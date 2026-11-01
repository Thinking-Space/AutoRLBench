import random
import os
from dataclasses import dataclass
from loguru import logger
from awm.gpt import GPTClient
from awm.tools import tools_jsonl_load, tools_jsonl_save, tools_robust_json_loads
from openai import OpenAI
import numpy as np
from tqdm import tqdm


from awm.prompts import (
    SCENARIO_CLASSIFICATION_SYSTEM_PROMPT,
    SCENARIO_CLASSIFICATION_USER_PROMPT_TEMPLATE,
    SCENARIO_GENERATION_SYSTEM_PROMPT,
    SCENARIO_GENERATION_USER_PROMPT_TEMPLATE,
    SCENARIO_FOCUSED_GENERATION_USER_PROMPT_TEMPLATE,
    SCENARIO_FOCUS_CATEGORIES,
    SCENARIO_DIVERSITY_CHECK_SYSTEM_PROMPT,
    SCENARIO_DIVERSITY_CHECK_USER_PROMPT_TEMPLATE,
    SCENARIO_CATEGORY_DIVERSITY_PROMPT,
)


@dataclass
class Config:
    input_path: str
    output_path: str
    target_count: int = 1000
    batch_size: int = 50
    num_parallel_requests: int = 5
    scenarios_per_request: int = 10
    num_few_shot: int = 8
    model: str = "your-llm-model-name"
    temperature: float = 1.0
    max_retries: int = 3
    diversity_check_batch: int = 80
    classify_only: bool = False
    skip_classification: bool = False
    embedding_similarity_threshold: float = 0.85
    embedding_warning_threshold: float = 0.75
    embedding_model: str = "text-embedding-3-large"
    embedding_batch_size: int = 50
    max_per_category: int = 80
    global_check_interval: int = 200
    max_stall_iterations: int = 10
    max_total_iterations: int = 1000
    resume: bool = False

    def pre_process(self):
        if self.resume:
            if not os.path.exists(self.output_path):
                if not os.path.exists(self.input_path):
                    raise FileNotFoundError(f"Resume mode: neither output ({self.output_path}) nor input ({self.input_path}) file found")
        else:
            if not os.path.exists(self.input_path):
                raise FileNotFoundError(f"Input file not found: {self.input_path}")

        output_dir = os.path.dirname(self.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        assert 'EMBEDDING_OPENAI_API_KEY' in os.environ, "EMBEDDING_OPENAI_API_KEY is required for embedding model"
        if 'EMBEDDING_OPENAI_BASE_URL' not in os.environ:
            logger.warning("EMBEDDING_OPENAI_BASE_URL is not set, using default OpenAI base url")

        if os.environ.get('AWM_SYN_OVERRIDE_MODEL'):
            self.model = os.environ.get('AWM_SYN_OVERRIDE_MODEL')
        
        assert self.model != "your-llm-model-name", "Please set the model name in the environment variable AWM_SYN_OVERRIDE_MODEL"


class ScenarioSelfInstruct:
    def __init__(self, args: Config):
        self.args = args
        self.client = GPTClient(timeout=600)
        self.existing_scenarios: list[dict] = []
        self.generated_scenarios: list[dict] = []
        self.all_names: set[str] = set()
        self.high_suitability: list[dict] = []
        self.medium_suitability: list[dict] = []
        self.low_suitability: list[dict] = []
        self.embedding_client = OpenAI(
            api_key=os.environ.get('EMBEDDING_OPENAI_API_KEY'),
            base_url=os.environ.get('EMBEDDING_OPENAI_BASE_URL', 'https://api.openai.com/v1'),
        )
        self.embeddings: np.ndarray | None = None
        self.embedding_dim: int = 3072
        self.category_counts: dict[str, int] = {}

    def load_existing(self):
        if self.args.resume and os.path.exists(self.args.output_path):
            load_path = self.args.output_path
            logger.info(f"Resume mode: loading from output file {load_path}")
        else:
            load_path = self.args.input_path
            if self.args.resume:
                logger.warning(f"Resume requested but {self.args.output_path} does not exist. Loading from {load_path}")

        self.existing_scenarios = tools_jsonl_load(load_path)
        logger.info(f"Loaded {len(self.existing_scenarios)} existing scenarios from {load_path}")

        for s in self.existing_scenarios:
            self.all_names.add(s['name'].lower().strip())

        classified_count = sum(1 for s in self.existing_scenarios if 'suitability_level' in s)
        logger.info(f"Already classified: {classified_count}/{len(self.existing_scenarios)}")

        return self.existing_scenarios

    def classify_scenarios(self, scenarios: list[dict]) -> list[dict]:
        to_classify = [s for s in scenarios if 'suitability_level' not in s]
        already_classified = [s for s in scenarios if 'suitability_level' in s]

        if not to_classify:
            logger.info("All scenarios already classified")
            return scenarios

        logger.info(f"Classifying {len(to_classify)} scenarios...")

        requests = []
        for s in to_classify:
            messages = [
                {"role": "system", "content": SCENARIO_CLASSIFICATION_SYSTEM_PROMPT},
                {"role": "user", "content": SCENARIO_CLASSIFICATION_USER_PROMPT_TEMPLATE.format(
                    name=s['name'],
                    description=s.get('description', 'No description')[:1500]
                )}
            ]
            requests.append({
                "messages": messages,
                "model": self.args.model,
                "max_tokens": 1000,
            })

        responses = self.client.batch_chat_completion(requests, progress_bar=True)

        classified = []
        for scenario, response in zip(to_classify, responses):
            try:
                result = tools_robust_json_loads(response)
                if isinstance(result, dict):
                    scenario['categories'] = result.get('categories', [])
                    scenario['suitability_score'] = result.get('suitability_score', 5)
                    scenario['suitability_level'] = result.get('suitability_level', 'medium')
                    scenario['suitability_reasoning'] = result.get('reasoning', '')
                    scenario['simulatable_features'] = result.get('simulatable_features', [])
                    scenario['non_simulatable_features'] = result.get('non_simulatable_features', [])
                else:
                    scenario['categories'] = []
                    scenario['suitability_score'] = 5
                    scenario['suitability_level'] = 'medium'
                    scenario['suitability_reasoning'] = 'Classification failed'
            except Exception as e:
                logger.warning(f"Failed to parse classification for {scenario['name']}: {e}")
                scenario['categories'] = []
                scenario['suitability_score'] = 5
                scenario['suitability_level'] = 'medium'
                scenario['suitability_reasoning'] = f'Classification error: {str(e)}'

            classified.append(scenario)

        all_classified = already_classified + classified

        high = sum(1 for s in all_classified if s.get('suitability_level') == 'high')
        medium = sum(1 for s in all_classified if s.get('suitability_level') == 'medium')
        low = sum(1 for s in all_classified if s.get('suitability_level') == 'low')
        logger.success(f"Classification complete: High={high}, Medium={medium}, Low={low}")

        return all_classified

    def filter_by_suitability(self):
        self.high_suitability = [s for s in self.existing_scenarios if s.get('suitability_level') == 'high']
        self.medium_suitability = [s for s in self.existing_scenarios if s.get('suitability_level') == 'medium']
        self.low_suitability = [s for s in self.existing_scenarios if s.get('suitability_level') == 'low']

        logger.info(f"Suitability distribution: High={len(self.high_suitability)}, Medium={len(self.medium_suitability)}, Low={len(self.low_suitability)}")

        if self.high_suitability:
            logger.debug(f"High suitability examples: {[s['name'] for s in self.high_suitability[:5]]}")
        if self.medium_suitability:
            logger.debug(f"Medium suitability examples: {[s['name'] for s in self.medium_suitability[:5]]}")
        if self.low_suitability:
            logger.debug(f"Low suitability examples: {[s['name'] for s in self.low_suitability[:5]]}")

    def _scenario_to_text(self, scenario: dict) -> str:
        name = scenario.get('name', '')
        desc = scenario.get('description', '')[:500]
        cats = ", ".join(scenario.get('categories', []))
        features = ", ".join(scenario.get('simulatable_features', [])[:5])
        return f"{name}: {desc} Categories: {cats}. Features: {features}"

    def compute_embeddings(self, scenarios: list[dict]) -> np.ndarray:
        if not scenarios:
            return np.array([]).reshape(0, self.embedding_dim)

        texts = [self._scenario_to_text(s) for s in scenarios]
        all_embeddings = []

        batch_size = self.args.embedding_batch_size
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            logger.debug(f"Computing embeddings for batch {i // batch_size + 1} ({len(batch_texts)} texts)")

            response = self.embedding_client.embeddings.create(
                input=batch_texts,
                model=self.args.embedding_model
            )

            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings)

    def initialize_embeddings(self):
        all_scenarios = self.existing_scenarios + self.generated_scenarios
        if not all_scenarios:
            self.embeddings = np.array([]).reshape(0, self.embedding_dim)
            return

        logger.info(f"Computing embeddings for {len(all_scenarios)} existing scenarios...")
        self.embeddings = self.compute_embeddings(all_scenarios)
        logger.success(f"Initialized embedding matrix: {self.embeddings.shape}")

        self._update_category_counts(all_scenarios)

    def _update_category_counts(self, scenarios: list[dict]):
        for s in scenarios:
            for cat in s.get('categories', []):
                self.category_counts[cat] = self.category_counts.get(cat, 0) + 1

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def max_similarity_to_pool(self, embedding: np.ndarray) -> tuple[float, int]:
        if self.embeddings is None or len(self.embeddings) == 0:
            return 0.0, -1

        norms = np.linalg.norm(self.embeddings, axis=1)
        norms = np.where(norms == 0, 1e-10, norms)

        embedding_norm = np.linalg.norm(embedding)
        if embedding_norm == 0:
            return 0.0, -1

        similarities = np.dot(self.embeddings, embedding) / (norms * embedding_norm)
        max_idx = int(np.argmax(similarities))
        max_sim = float(similarities[max_idx])

        return max_sim, max_idx

    def check_embedding_diversity(self, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []

        logger.info(f"Computing embeddings for {len(candidates)} candidates...")
        candidate_embeddings = self.compute_embeddings(candidates)

        kept = []
        kept_embeddings = []

        all_scenarios = self.existing_scenarios + self.generated_scenarios

        for i, (candidate, emb) in enumerate(zip(candidates, candidate_embeddings)):
            max_sim, most_similar_idx = self.max_similarity_to_pool(emb)

            batch_max_sim = 0.0
            if kept_embeddings:
                kept_emb_array = np.array(kept_embeddings)
                for kept_emb in kept_emb_array:
                    sim = self.cosine_similarity(emb, kept_emb)
                    batch_max_sim = max(batch_max_sim, sim)

            overall_max_sim = max(max_sim, batch_max_sim)

            if overall_max_sim > self.args.embedding_similarity_threshold:
                if most_similar_idx >= 0 and most_similar_idx < len(all_scenarios):
                    similar_name = all_scenarios[most_similar_idx].get('name', 'Unknown')
                else:
                    similar_name = "batch candidate"
                logger.debug(f"REJECT (sim={overall_max_sim:.3f}): {candidate['name']} ~= {similar_name}")
                continue

            if overall_max_sim > self.args.embedding_warning_threshold:
                categories = candidate.get('categories', [])
                over_limit = False
                for cat in categories:
                    if self.category_counts.get(cat, 0) >= self.args.max_per_category:
                        logger.debug(f"REJECT (category limit): {candidate['name']} - {cat} has {self.category_counts[cat]} scenarios")
                        over_limit = True
                        break
                if over_limit:
                    continue

            kept.append(candidate)
            kept_embeddings.append(emb)

            if overall_max_sim > self.args.embedding_warning_threshold:
                logger.debug(f"ACCEPT (sim={overall_max_sim:.3f}, borderline): {candidate['name']}")
            else:
                logger.debug(f"ACCEPT (sim={overall_max_sim:.3f}): {candidate['name']}")

        logger.info(f"Embedding diversity check: kept {len(kept)}/{len(candidates)} candidates")
        return kept

    def add_to_embedding_pool(self, scenarios: list[dict]):
        if not scenarios:
            return

        new_embeddings = self.compute_embeddings(scenarios)

        if self.embeddings is None or len(self.embeddings) == 0:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        self._update_category_counts(scenarios)

        logger.debug(f"Embedding pool updated: {self.embeddings.shape}")

    def run_global_diversity_check(self) -> dict:
        if self.embeddings is None or len(self.embeddings) < 10:
            return {"status": "not_enough_data"}

        n = len(self.embeddings)
        logger.info(f"Running global diversity check on {n} scenarios...")

        sample_size = min(200, n)
        indices = random.sample(range(n), sample_size)
        sample_embeddings = self.embeddings[indices]

        norms = np.linalg.norm(sample_embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        normalized = sample_embeddings / norms
        similarity_matrix = np.dot(normalized, normalized.T)

        upper_tri = similarity_matrix[np.triu_indices(sample_size, k=1)]

        stats = {
            "num_scenarios": n,
            "sample_size": sample_size,
            "mean_similarity": float(np.mean(upper_tri)),
            "max_similarity": float(np.max(upper_tri)),
            "min_similarity": float(np.min(upper_tri)),
            "std_similarity": float(np.std(upper_tri)),
            "high_similarity_pairs": int(np.sum(upper_tri > self.args.embedding_similarity_threshold)),
            "category_distribution": dict(sorted(self.category_counts.items(), key=lambda x: -x[1])),
        }

        over_represented = [cat for cat, count in self.category_counts.items()
                          if count > self.args.max_per_category]
        stats["over_represented_categories"] = over_represented

        logger.info(f"Global diversity stats:")
        logger.info(f"  Mean similarity: {stats['mean_similarity']:.3f}")
        logger.info(f"  Max similarity: {stats['max_similarity']:.3f}")
        logger.info(f"  High similarity pairs (>{self.args.embedding_similarity_threshold}): {stats['high_similarity_pairs']}")

        if over_represented:
            logger.warning(f"  Over-represented categories: {over_represented}")

        top_cats = list(self.category_counts.items())[:10]
        logger.info(f"  Top categories: {top_cats}")

        return stats

    def get_few_shot_examples(self, num_examples: int, prioritize_diverse: bool = True) -> list[dict]:
        pool = self.high_suitability

        if not pool:
            logger.warning("No suitable scenarios found, using all existing scenarios")
            pool = self.existing_scenarios

        if not prioritize_diverse or len(pool) <= num_examples:
            return random.sample(pool, min(num_examples, len(pool)))

        categorized: dict[str, list[dict]] = {}
        uncategorized = []

        for scenario in pool:
            categories = scenario.get('categories', [])
            if categories:
                primary_cat = categories[0]
                if primary_cat not in categorized:
                    categorized[primary_cat] = []
                categorized[primary_cat].append(scenario)
            else:
                uncategorized.append(scenario)

        selected = []
        if categorized:
            per_category = max(1, num_examples // len(categorized))

            for cat, scenarios in categorized.items():
                if scenarios and len(selected) < num_examples:
                    to_take = min(per_category, len(scenarios), num_examples - len(selected))
                    selected.extend(random.sample(scenarios, to_take))

        remaining = num_examples - len(selected)
        if remaining > 0:
            remaining_pool = uncategorized + [s for s in pool if s not in selected]
            if remaining_pool:
                selected.extend(random.sample(remaining_pool, min(remaining, len(remaining_pool))))

        random.shuffle(selected)
        return selected[:num_examples]

    def format_examples(self, examples: list[dict]) -> str:
        formatted = []
        for i, ex in enumerate(examples, 1):
            formatted.append(f'{i}. {{"name": "{ex["name"]}", "description": "{ex["description"][:500]}..."}}')
        return "\n\n".join(formatted)

    def _build_generation_request(
        self,
        examples: list[dict],
        num_to_generate: int,
        focus_categories: list[str] | None = None,
        avoid_scenarios: list[dict] | None = None
    ) -> dict:
        examples_text = self.format_examples(examples)

        if avoid_scenarios:
            examples_text += "\n\n## Also avoid generating scenarios similar to these recently added ones:\n"
            examples_text += self.format_examples(avoid_scenarios[:10])

        if focus_categories:
            focus_text = "\n".join([f"- {cat}" for cat in focus_categories])
            user_content = SCENARIO_FOCUSED_GENERATION_USER_PROMPT_TEMPLATE.format(
                num_examples=len(examples),
                examples=examples_text,
                focus_categories=focus_text,
                num_to_generate=num_to_generate
            )
        else:
            user_content = SCENARIO_GENERATION_USER_PROMPT_TEMPLATE.format(
                num_examples=len(examples),
                examples=examples_text,
                num_to_generate=num_to_generate
            )

        return {
            "messages": [
                {"role": "system", "content": SCENARIO_GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            "model": self.args.model,
            "temperature": self.args.temperature,
            "max_tokens": 16000,
        }

    def _parse_generation_response(self, response: str) -> list[dict]:
        try:
            parsed = tools_robust_json_loads(response)

            if not isinstance(parsed, list):
                return []

            valid = []
            for item in parsed:
                if isinstance(item, dict) and 'name' in item and 'description' in item:
                    name_lower = item['name'].lower().strip()

                    if name_lower not in self.all_names:
                        valid.append(item)

            return valid
        except Exception as e:
            logger.warning(f"Failed to parse response: {e}")
            return []

    def generate_batch(self, num_to_generate: int) -> list[dict]:
        num_requests = self.args.num_parallel_requests
        per_request = self.args.scenarios_per_request

        requests = []

        avoid_scenarios = self.generated_scenarios[-20:] if self.generated_scenarios else []

        shuffled_categories = SCENARIO_FOCUS_CATEGORIES.copy()
        random.shuffle(shuffled_categories)

        for i in range(num_requests):
            examples = self.get_few_shot_examples(
                self.args.num_few_shot,
                prioritize_diverse=True
            )
            random.shuffle(examples)

            focus_idx = i % len(shuffled_categories)
            focus_cats = []
            for j in range(3):
                cat_idx = (focus_idx + j) % len(shuffled_categories)
                focus_cats.extend(shuffled_categories[cat_idx])

            use_focus = random.random() > 0.3

            request = self._build_generation_request(
                examples=examples,
                num_to_generate=per_request,
                focus_categories=focus_cats if use_focus else None,
                avoid_scenarios=avoid_scenarios
            )
            requests.append(request)

        logger.info(f"Sending {num_requests} parallel requests, each generating {per_request} scenarios...")

        responses = self.client.batch_chat_completion(
            requests=requests,
            progress_bar=True,
        )

        all_valid = []
        seen_names = set()

        for i, response in enumerate(responses):
            parsed = self._parse_generation_response(response)
            logger.debug(f"Request {i+1}: got {len(parsed)} valid scenarios")

            for item in parsed:
                name_lower = item['name'].lower().strip()
                if name_lower not in seen_names:
                    seen_names.add(name_lower)
                    all_valid.append(item)

        logger.success(f"Generated {len(all_valid)} unique valid scenarios from {num_requests} parallel requests")

        return all_valid

    def check_diversity(self, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []

        existing_sample = self.existing_scenarios + self.generated_scenarios
        if len(existing_sample) > 100:
            existing_sample = random.sample(existing_sample, 100)

        existing_text = "\n".join([
            f"- {s['name']}: {s['description'][:200]}..."
            for s in existing_sample
        ])

        candidates_text = "\n".join([
            f"{i}. {c['name']}: {c['description'][:200]}..."
            for i, c in enumerate(candidates)
        ])

        messages = [
            {"role": "system", "content": SCENARIO_DIVERSITY_CHECK_SYSTEM_PROMPT},
            {"role": "user", "content": SCENARIO_DIVERSITY_CHECK_USER_PROMPT_TEMPLATE.format(
                existing_scenarios=existing_text,
                new_candidates=candidates_text
            )}
        ]

        logger.info(f"Checking diversity for {len(candidates)} candidates...")

        try:
            response = self.client.chat_completion(
                messages=messages,
                model=self.args.model,
                max_tokens=8000,
            )

            result = tools_robust_json_loads(response)

            if isinstance(result, dict) and 'decisions' in result:
                kept = []
                for decision in result['decisions']:
                    idx = decision.get('index', -1)
                    if 0 <= idx < len(candidates) and decision.get('decision') == 'keep':
                        kept.append(candidates[idx])
                    elif decision.get('decision') == 'reject':
                        logger.debug(f"Rejected: {decision.get('name')} - {decision.get('reason')}")

                logger.info(f"Diversity check: kept {len(kept)}/{len(candidates)} candidates")
                return kept

        except Exception as e:
            logger.error(f"Diversity check failed: {e}")
            return candidates

        return candidates

    def analyze_category_distribution(self) -> dict:
        all_scenarios = self.existing_scenarios + self.generated_scenarios

        scenarios_text = "\n".join([
            f"- {s['name']}: {s['description'][:150]}..."
            for s in all_scenarios[-200:]
        ])

        messages = [
            {"role": "system", "content": "You are an expert at categorizing website/app scenarios."},
            {"role": "user", "content": SCENARIO_CATEGORY_DIVERSITY_PROMPT.format(scenarios=scenarios_text)}
        ]

        try:
            response = self.client.chat_completion(
                messages=messages,
                model=self.args.model,
                max_tokens=4000,
            )

            result = tools_robust_json_loads(response)
            if isinstance(result, dict):
                logger.info(f"Category analysis: {result.get('underrepresented', [])}")
                return result

        except Exception as e:
            logger.error(f"Category analysis failed: {e}")

        return {}

    def add_scenarios(self, scenarios: list[dict]):
        for s in scenarios:
            name_lower = s['name'].lower().strip()

            if name_lower not in self.all_names:
                self.generated_scenarios.append(s)
                self.all_names.add(name_lower)

    def save_progress(self):
        all_scenarios = self.existing_scenarios + self.generated_scenarios
        tools_jsonl_save(all_scenarios, self.args.output_path)
        logger.info(f"Saved {len(all_scenarios)} scenarios to {self.args.output_path}")

    def run(self):
        self.load_existing()

        if not self.args.skip_classification and not self.args.resume:
            logger.info("\n" + "="*60)
            logger.info("Step 1: Classifying existing scenarios by suitability...")
            logger.info("="*60)

            self.existing_scenarios = self.classify_scenarios(self.existing_scenarios)

            tools_jsonl_save(self.existing_scenarios, self.args.input_path)
            logger.success(f"Saved classified scenarios to {self.args.input_path}")
        elif self.args.resume:
            logger.info("Resume mode: skipping classification step (already done)")

        self.filter_by_suitability()

        if self.args.classify_only:
            logger.success("Classification complete (--classify_only mode)")
            self._print_suitability_report()
            return

        suitable_pool = self.high_suitability

        logger.info("\n" + "="*60)
        logger.info("Step 2: Initializing embedding-based diversity system...")
        logger.info("="*60)
        self.initialize_embeddings()

        logger.info("\n" + "="*60)
        logger.info("Step 3: Generating new scenarios via self-instruct...")
        logger.info("="*60)

        current_count = len(self.existing_scenarios) + len(self.generated_scenarios)
        target = self.args.target_count

        expected_per_iteration = self.args.num_parallel_requests * self.args.scenarios_per_request

        logger.info(f"Starting self-instruct: {current_count} -> {target} scenarios")
        logger.info(f"Parallel requests: {self.args.num_parallel_requests}")
        logger.info(f"Scenarios per request: {self.args.scenarios_per_request}")
        logger.info(f"Expected per iteration: ~{expected_per_iteration}")
        logger.info(f"Temperature: {self.args.temperature}")
        logger.info(f"Embedding similarity threshold: {self.args.embedding_similarity_threshold}")
        logger.info(f"Max per category: {self.args.max_per_category}")
        logger.info(f"Max stall iterations: {self.args.max_stall_iterations}")
        logger.info(f"Max total iterations: {self.args.max_total_iterations}")

        iteration = 0
        pending_candidates = []
        last_global_check = current_count
        stall_count = 0
        last_count = current_count

        pbar = tqdm(
            total=target,
            initial=current_count,
            desc="Generating scenarios",
            unit="scenarios",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )

        while current_count < target:
            iteration += 1
            remaining = target - current_count

            if iteration > self.args.max_total_iterations:
                logger.error(f"Reached max iterations ({self.args.max_total_iterations}). Stopping.")
                logger.error(f"Generated {current_count}/{target} scenarios.")
                break

            if stall_count >= self.args.max_stall_iterations:
                logger.warning(f"Stalled for {stall_count} iterations. Adjusting strategy...")

                old_threshold = self.args.embedding_similarity_threshold
                self.args.embedding_similarity_threshold = min(0.95, old_threshold + 0.05)
                logger.info(f"Increased similarity threshold: {old_threshold:.2f} -> {self.args.embedding_similarity_threshold:.2f}")

                self.args.max_per_category = int(self.args.max_per_category * 1.2)
                logger.info(f"Increased max_per_category to {self.args.max_per_category}")

                stall_count = 0

            if iteration % 5 == 1:
                logger.debug(f"Iteration {iteration}: Current={current_count}, Remaining={remaining}")

            if iteration % 5 == 1 and current_count > 100:
                category_info = self.analyze_category_distribution()
                if category_info.get('underrepresented'):
                    logger.debug(f"Underrepresented: {category_info['underrepresented'][:3]}")

            new_batch = self.generate_batch(expected_per_iteration)
            if not new_batch:
                logger.warning("Generation returned no results, retrying...")
                stall_count += 1
                continue

            pending_candidates.extend(new_batch)

            if len(pending_candidates) >= self.args.diversity_check_batch or remaining <= expected_per_iteration:
                kept = self.check_embedding_diversity(pending_candidates)

                if kept:
                    self.add_scenarios(kept)
                    self.add_to_embedding_pool(kept)

                pending_candidates = []

                prev_count = current_count
                current_count = len(self.existing_scenarios) + len(self.generated_scenarios)

                pbar.update(current_count - prev_count)

                if current_count == last_count:
                    stall_count += 1
                else:
                    stall_count = 0
                    last_count = current_count

                if current_count != prev_count:
                    self.save_progress()

                if current_count - last_global_check >= self.args.global_check_interval:
                    pbar.set_postfix_str("Global check...")
                    self.run_global_diversity_check()
                    last_global_check = current_count
                    pbar.set_postfix_str("")

        pbar.close()

        logger.info("\n" + "="*60)
        logger.info("Final global diversity check...")
        logger.info("="*60)
        final_stats = self.run_global_diversity_check()

        self.save_progress()

        logger.success(f"\n{'='*60}")
        if current_count >= target:
            logger.success(f"Self-instruct complete!")
        else:
            logger.warning(f"Self-instruct stopped early (reached iteration limit)")
        logger.success(f"Original: {len(self.existing_scenarios)}")
        logger.success(f"Generated: {len(self.generated_scenarios)}")
        logger.success(f"Total: {len(self.existing_scenarios) + len(self.generated_scenarios)}")
        logger.success(f"Target: {target}")
        logger.success(f"Iterations: {iteration}")
        logger.success(f"Output: {self.args.output_path}")
        if isinstance(final_stats.get('mean_similarity'), float):
            logger.success(f"Mean similarity: {final_stats['mean_similarity']:.3f}")
        logger.success(f"{'='*60}")

    def _print_suitability_report(self):
        logger.info("\n" + "="*60)
        logger.info("SUITABILITY REPORT")
        logger.info("="*60)

        logger.info(f"\nHIGH Suitability ({len(self.high_suitability)} scenarios):")
        for s in self.high_suitability[:10]:
            cats = ", ".join(s.get('categories', [])[:3])
            logger.info(f"  - {s['name']}: [{cats}]")
        if len(self.high_suitability) > 10:
            logger.info(f"  ... and {len(self.high_suitability) - 10} more")

        logger.info(f"\nMEDIUM Suitability ({len(self.medium_suitability)} scenarios):")
        for s in self.medium_suitability[:10]:
            cats = ", ".join(s.get('categories', [])[:3])
            reason = s.get('suitability_reasoning', '')[:80]
            logger.info(f"  - {s['name']}: [{cats}] - {reason}")
        if len(self.medium_suitability) > 10:
            logger.info(f"  ... and {len(self.medium_suitability) - 10} more")

        logger.info(f"\nLOW Suitability ({len(self.low_suitability)} scenarios) - NOT suitable for env synthesis:")
        for s in self.low_suitability:
            reason = s.get('suitability_reasoning', '')[:80]
            logger.info(f"  - {s['name']}: {reason}")

        logger.info("\n" + "-"*40)
        logger.info("Category Distribution (High + Medium suitability only):")
        category_counts: dict[str, int] = {}
        for s in self.high_suitability + self.medium_suitability:
            for cat in s.get('categories', []):
                category_counts[cat] = category_counts.get(cat, 0) + 1

        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            logger.info(f"  {cat}: {count}")


def run(config: Config):
    instructor = ScenarioSelfInstruct(config)
    instructor.run()
