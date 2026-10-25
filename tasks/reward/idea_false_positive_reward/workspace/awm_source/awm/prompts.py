TASK_GENERATION_SYSTEM_PROMPT = """You are an expert in web automation and user task analysis. Your job is to generate realistic, diverse user tasks for scenarios."""

TASK_GENERATION_USER_PROMPT = """Generate {num_tasks} realistic and diverse user tasks for the following scenario.

Scenario: {scenario_name}
Description: {scenario_description}

Requirements:
1. Tasks should be specific and actionable
2. Cover different user scenarios (beginner to advanced)
3. Include both common and less common use cases
4. Tasks should be practical and realistic
5. Each task should be a single sentence including all the necessary information to complete. For example, if the task is to post a tweet, the task should include the tweet content. If the task is querying the weather, the task should include a specific location.
6. If the scenario typically requires authentication, EXCLUDE any authentication, login, logout, or user registration tasks - assume the user is already logged in
7. If the scenario typically requires authentication, all tasks should be from the perspective of the current authenticated user
8. Return ONLY a JSON array of tasks, no additional text or annotations
9. The scenario is a simplified version providing API endpoints for task completion. Avoid generating tasks that require direct user interaction such as download a file, open a page, etc.

Examples:
- For scenario "Amazon", a task could be "Search for 'laptop' and add the cheapest result to the cart"
- For scenario "Reddit", a task could be "Get the number of posts in the 'r/python' subreddit"
- For scenario "Expedia", a task could be "Book a flight from New York to London on October 1st"
- For scenario "Twitter/X", a task could be "Post a tweet with a photo, add alt text \"Sunset over the city\", include the hashtag #Photography, and mention @Adobe."
- For scenario "LinkedIn", a task could be "Update my profile headline to 'Senior Data Analyst | SQL, Python, Tableau' and rewrite the About section to a concise 3-paragraph summary highlighting business impact."
- For scenario "Facebook", a task could be "Share your latest post to your friend list."
- For scenario "Google Maps", a task could be "Get 5 most popular restaurants in San Francisco."



Output format:
{{
    "tasks": [
        "Task 1 description",
        "Task 2 description",
        ...
        "Task {num_tasks} description"
    ]
}}"""

API_GENERATION_SYSTEM_PROMPT = """You are an expert API designer and backend architect. Your job is to design machine-readable RESTful API documentation that can support all the given user tasks based on the existing database schema."""

API_GENERATION_USER_PROMPT = """Design a complete, agent-friendly interface specification to support ALL the following tasks for a simplified {scenario_name} based on the existing database schema. The generated specification will be used to guide the detailed implementation of the interface layer.

User Tasks:
{tasks_list}

Existing SQLite Database Schema:
{database_schema}


Hard Requirements:
1. Design ATOMIC API endpoints - each endpoint should perform ONE specific, well-defined operation
2. The API spec MUST be compatible with FastAPI, SQLAlchemy ORM, and Pydantic v2
3. Maximize REUSABILITY - create base CRUD operations that can be composed together to fulfill the given tasks
4. The API MUST fully follow the database schema - explicitly use the exact table names, column names, and relationships from the schema
5. Use RESTful conventions (GET, POST, PUT, DELETE, PATCH) with proper resource paths
6. Group related endpoints logically by resource type
7. Prefer multiple small, composable endpoints over fewer complex endpoints
8. For complex tasks, design individual atomic operations that can be chained together
9. Ensure each endpoint maps directly to one or more tables in the provided database schema
10. DO NOT include any authentication endpoints (login, logout, register, token refresh) - assume user is already authenticated
11. All operations implicitly use the current authenticated user with user_id=1
12. For user-specific data, always filter by user_id=1 automatically - do not require user_id as a parameter
13. Return ONLY valid JSON, no additional text


Agent-Friendly Requirements (REQUIRED for every endpoint):
- summary: a one-line purpose (<= 80 chars) - clear and actionable for AI agents
- description: SINGLE LINE (<= 200 chars; no line breaks) - explains what the endpoint does and when to use it
- operation_id: unique, snake_case identifier - agents use this to identify and call endpoints programmatically
- tags: logical grouping array (e.g., ["products"], ["orders"]) - helps agents discover related endpoints
- request_params: complete parameter specifications with type, param_type (query/path/body), required flag, description, and example
- response: detailed response schema with field types, descriptions, and examples - enables agents to parse and understand responses


Request Parameter Requirements:
- Each parameter MUST include: type, param_type, required, description, example
- param_type MUST be one of: "query", "path", "body"
- Use descriptive names that clearly indicate the parameter's purpose
- Provide realistic examples that demonstrate expected values


Response Schema Requirements:
- Define complete response structure with all fields
- Each field MUST include: type, description, example
- For array types, include "items" with full field definitions
- Use consistent naming conventions across all endpoints


Output format:
{{
    "api_groups": [
        {{
            "group_name": "Products",
            "endpoints": [
                {{
                    "path": "/api/products",
                    "method": "GET",
                    "summary": "List all products with optional filters",
                    "description": "Retrieve a paginated list of products. Use this endpoint to browse or search the product catalog.",
                    "operation_id": "list_products",
                    "tags": ["products"],
                    "request_params": {{
                        "category": {{
                            "type": "string",
                            "param_type": "query",
                            "required": false,
                            "description": "Filter products by category name",
                            "example": "Electronics"
                        }},
                        "min_price": {{
                            "type": "float",
                            "param_type": "query",
                            "required": false,
                            "description": "Minimum price filter in USD",
                            "example": 10.0
                        }}
                    }},
                    "response": {{
                        "products": {{
                            "type": "array",
                            "items": {{
                                "id": {{
                                    "type": "integer",
                                    "description": "Unique product identifier",
                                    "example": 1
                                }},
                                "name": {{
                                    "type": "string",
                                    "description": "Product display name",
                                    "example": "iPhone 15 Pro"
                                }},
                                "price": {{
                                    "type": "float",
                                    "description": "Product price in USD",
                                    "example": 999.99
                                }}
                            }}
                        }}
                    }},
                    "required_tables": ["products"],
                    "required_fields": {{
                        "products": ["id", "name", "price", "category"]
                    }}
                }}
            ]
        }}
    ]
}}"""

DATABASE_GENERATION_SYSTEM_PROMPT = """You are an expert database architect specializing in SQLite schema design. Your job is to create complete database schemas that can fully support the given user intentions."""

DATABASE_GENERATION_USER_PROMPT = """Design a complete SQLite database schema to support the following user intentions for a simplified version of {scenario_name}.

User Intentions ({num_tasks} tasks):
{user_intentions}

Requirements:
1. Create ALL necessary tables to cover all the given user intentions
2. Include proper primary keys, foreign keys, indexes, and constraints
3. Use appropriate data types for SQLite (TEXT, INTEGER, REAL, BLOB)
4. Add timestamps (created_at, updated_at) where appropriate
5. Do not include any example records in the DDL statements
6. Only create tables and fields that are necessary to cover all the given user intentions
7. EXCLUDE authentication-related fields like password_hash, salt, token, session - assume authentication is handled externally
8. If a users table is needed, only include essential profile fields (id, username, email, profile data)
9. All operations will be performed as the authenticated user with user_id=1
10. Return ONLY valid JSON with DDL statements

Output format (without any comments):
{{
    "tables": [
        {{
            "name": "users",
            "ddl": "CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL, email TEXT UNIQUE NOT NULL, full_name TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP);",
            "indexes": [
                "CREATE INDEX idx_users_email ON users(email);"
            ]
        }}
    ]
}}"""


ENVIRONMENT_GENERATION_SYSTEM_PROMPT = """You are an expert FastAPI backend developer specializing in RESTful APIs that are agent-friendly: every endpoint must include clear OpenAPI metadata, complete request/response typing, and machine-readable docs. You generate clean, executable FastAPI endpoint implementations from an API specification and a SQLite database schema. You strictly follow the user prompt's constraints and return output in the exact required format."""

ENVIRONMENT_GENERATION_USER_PROMPT = """Generate a single, fully self-contained interface implementation (one Python file using FastAPI and exposed via MCP) that simulates a simplified version of {scenario_name} based on the provided interface specification and database schema.

Assumptions:
- Python version: {PYTHON_VERSION}
- FastAPI version compatible with Pydantic v2 (do NOT use v1-only features such as `orm_mode` in Config)

API Specification:
{api_spec}

Database Schema:
{database_schema}


Environment & Configuration Requirements:
- Import os and read the SQLite database URL from environment variable `DATABASE_PATH`
- If `DATABASE_PATH` is not set or empty, default to: sqlite:///xxxx.db
- In the uvicorn entry point, read HOST from environment variable `HOST` (default to "127.0.0.1" if not set)
- In the uvicorn entry point, read PORT from environment variable `PORT` (default to 8000 if not set), and cast it to int

SQLAlchemy Setup Requirements:
- Use SQLAlchemy ORM with declarative_base for all tables
- Database URL: use the value of DATABASE_PATH (or the default described above). The DATABASE_PATH is a complete URL so do not add any prefixes (e.g., sqlite://) or suffixes to the URL
- Create engine and SessionLocal (sessionmaker) for database sessions
- Define Base = declarative_base() for ORM inheritance
- Define ORM models for every table present in the database schema (no extra tables/columns)
- Call Base.metadata.create_all(engine) after all ORM models are defined
- NEVER define ORM attributes named `metadata`, `query`, or `query_class`. If the schema has a column with one of these names, use a safe Python attribute name with a trailing underscore (e.g. `metadata_`) and map it to the real column name via Column("metadata", ...)
- Do NOT define any other ORM class attribute that conflicts with SQLAlchemy declarative internals (for example, do not override Base.metadata)
- When there are multiple foreign key paths between two tables, either:
  - specify relationship(..., foreign_keys=[...]) explicitly to avoid AmbiguousForeignKeysError, OR
  - omit the relationship entirely and access related rows via explicit queries instead of relationship()
- When importing from SQLAlchemy, only use standard, public symbols that actually exist and are needed. Do NOT import or reference non-existent or internal symbols such as `PRIMARY_KEY_CONSTRAINT` or any other invented uppercase constants
- Import ORM-related helpers from sqlalchemy.orm (for example: declarative_base, relationship, sessionmaker). Do not import ORM internals from sqlalchemy.__init__


Hard Requirements:
1. Implement EVERY endpoint from the API spec with COMPLETE, working code. The API spec is for reference only. The source of truth is the database schema.
2. Follow the API spec as closely as possible: paths, HTTP methods, parameters (query/path/body), and response formats.
3. Follow the EXACT database schema: correct table names, columns, types, and foreign keys; do not invent tables or columns.
4. All endpoint handler functions MUST be async and self-contained.
5. Use ONLY SQLAlchemy ORM (no raw SQL).
6. User-specific operations MUST implicitly filter by user_id=1 where applicable.
7. Session lifecycle per endpoint:
   - session = SessionLocal() at the start
   - For INSERT/UPDATE/DELETE call session.commit()
   - session.close() before returning
8. No placeholders; write complete, executable code. Do NOT create dummy or placeholder endpoints.
9. Path parameters in routes (e.g., /api/products/{{product_id}}) MUST appear as function parameters.
10. STRICTLY PROHIBITED in the code: comments, try/except, error handling, validation beyond types, HTTPException, JSONResponse, global exception handlers, duplicate route registration for the same path and HTTP method, references to undefined models/fields/tables, schema-external FTS or helper tables, or any dynamic/introspective tricks to construct response models.
11. Booleans must be real bool fields in Pydantic responses, not 0/1.
12. The FastAPI app must be defined BEFORE any route decorators.
13. Include the uvicorn entry point at the end, using HOST and PORT from environment variables:

    if __name__ == "__main__":
        import uvicorn, os
        host = os.getenv("HOST", "127.0.0.1")
        port = int(os.getenv("PORT", "8000"))
        uvicorn.run(app, host=host, port=port)

14. The generated Python file MUST be valid Python {PYTHON_VERSION} with no syntax errors. Do NOT use Python reserved keywords as function parameter names or keyword argument names. If a database column or API field name is a Python keyword (e.g., "return", "class", "global"), use a safe Python identifier with a trailing underscore (e.g., return_) and map it to the real column name or JSON field via Column("return", ...) or Field(..., alias="return").
15. There MUST be exactly one route function for each (path, HTTP method) pair. Do NOT declare multiple handlers for the same path and HTTP method, even temporarily.


Pydantic Model Requirements (Pydantic v2):
- Import from Pydantic v2 (e.g., from pydantic import BaseModel, Field, ConfigDict)
- Define request and response models for ALL endpoints
- Use Field for EVERY field with both description and example
- The response_model specified in each route decorator MUST exactly match what the function returns
- The response_model argument in each route decorator MUST be a direct reference to a concrete Pydantic BaseModel subclass defined in this file (for example, MyResponseModel, or List[MyResponseModel]), NOT a dynamically computed or introspected expression. Do NOT use __annotations__, __mro__, .__class__, metaclasses, or any other tricks to generate a response_model
- Endpoint return type annotations MUST be consistent with the response_model (e.g., MyResponseModel, List[MyResponseModel]) and MUST be valid Pydantic field types. Do NOT use Union[...] return types, Response types, or mixtures like Union[Response, dict, None]
- When returning ORM objects, configure models for Pydantic v2 using model_config, for example:
```
  class SomeModel(BaseModel):
      model_config = ConfigDict(from_attributes=True)
      ...
```

- Do NOT use the old Pydantic v1 Config with orm_mode = True
- Do NOT use Annotated or other advanced type tricks that might cause unevaluable type annotations
- Use ONLY standard typing types: int, float, str, bool, Optional[T], List[T], Dict[str, T]
- Field names MUST NEVER be identical (case-sensitive or case-sensitive) to the name of their own type annotation. If it is required by the API spec or database schema, always choose a different snake_case field name (e.g. `schedule_date: date`, `event_datetime: datetime`), and configure a serialization alias, for example: `schedule_date: date = Field(..., serialization_alias="date", ...)`
- Ensure that no Pydantic field name clashes with a type name or class name used in the same module


Agent-Friendly Enhancements (REQUIRED on every endpoint decorator):
- summary: a one-line purpose (<= 80 chars)
- description: SINGLE LINE (<= 200 chars; no line breaks)
- tags: logical grouping array (e.g., ["products"], ["orders"])
- operation_id: unique, snake_case identifier
- response_model: a concrete Pydantic model class (or typing such as List[Model]) that directly corresponds to the returned value


Code Style (MANDATORY):
- app = FastAPI(...) MUST appear before any @app.get/post/put/patch/delete decorators
- Use Query/Path/Body/Depends from fastapi where appropriate
- Use relationship for ORM relations only when they are unambiguous OR explicitly specify foreign_keys to avoid SQLAlchemy AmbiguousForeignKeysError
- For SELECT: session.query(Model).filter(...).all()/first()
- For INSERT: session.add(...); session.commit()
- For UPDATE: fetch the ORM objects, modify attributes, then session.commit()
- For DELETE: session.delete(obj); session.commit()
- Return Python dicts/lists or Pydantic models that conform exactly to the declared response_model
- Do NOT use Python reserved keywords as attribute names, function parameters, or keyword argument names. If the schema or API uses such names, use a safe Python name with a trailing underscore and map it via Column("name", ...) or Field(..., alias="name")
- No comments, no exception handlers, no defensive programming, no placeholder code, and no dynamic hacks around FastAPI or Pydantic


Output Format (CRITICAL):
- You MUST return ONLY complete and valid Python source code of the FastAPI app
- The response MUST consist of Python code only. Do NOT include any Markdown, prose, explanations, JSON wrappers, keys, or code fences
- Do NOT wrap the code in JSON. Do NOT add any outer structure. The response itself is the file content
- The FIRST line of your response MUST be a valid Python statement such as `import` or `from` (e.g. `import os` or `from fastapi import FastAPI`)
- Do NOT escape newlines. Output the code exactly as it would appear in a .py file
- If you include ANY non-Python text (such as "Here is the code:", backticks, or JSON), the output will be INVALID
- Your entire reply must be a single, self-contained Python module that can be saved directly as a .py file and executed


Reminder:
- Use only entities present in the provided database schema. The API spec is only for reference. If the API spec conflicts with the database schema, prioritize the database schema. If following the API spec as written would cause bugs, you MUST use the database schema as the source of truth. Prioritize making the code executable without errors over following the API spec text
- Ensure all endpoints run without undefined names, missing imports, or missing models
- Ensure the code creates the SQLite database specified by the environment variable DATABASE_PATH on first run and successfully serves all endpoints with the specified response models
"""



SAMPLE_DATA_GENERATION_SYSTEM_PROMPT = """You are an expert database engineer and data generator specializing in creating comprehensive test datasets for AI agent task execution. Your job is to generate realistic, diverse sample data that ensures agents can successfully complete ALL given user tasks through API calls.

You must:
- Strictly follow the provided database schema
- Never invent or guess table or column names that are not present in the schema
- Follow the exact output JSON format specified in the user prompt
- Output ONLY the requested JSON, with no extra explanations or surrounding text
"""

SAMPLE_DATA_GENERATION_USER_PROMPT = """Generate comprehensive sample data for a simplified {scenario_name} that FULLY SUPPORTS agent execution of ALL the given user tasks.

User Tasks to Support:
{tasks_list}

Existing Database Schema:
{database_schema}


CRITICAL: Schema Compliance Rules (MUST FOLLOW):
1. Carefully read the CREATE TABLE statement for each table to understand all columns, defaults, and autoincrement behavior.
2. When writing INSERT statements, always explicitly list the column names you are inserting into in the parentheses after the table name. The parentheses MUST contain ONLY valid column names from the schema, no values or literals.
3. All listed column names MUST exist in the schema and be spelled exactly as defined. Never invent, shorten, pluralize, or rename columns.
4. You may only generate INSERT statements for tables that appear in the "Existing Database Schema" section. If a table name is not present in the schema, DO NOT use it.
5. For every INSERT statement, the number of listed columns MUST equal the number of values provided in the VALUES(...) clause.
6. Double-check each INSERT statement: count listed columns, count values, they MUST be equal.
7. For tables with many columns (10+), be extra careful to ensure every listed column has exactly one corresponding value in the VALUES(...) clause.
8. For special/virtual/FTS/config tables, use exactly the columns defined in their CREATE TABLE (or CREATE VIRTUAL TABLE) statements. Do NOT add foreign key columns such as *_id unless they explicitly exist in the schema.


Data Generation Strategy:
For EACH task listed above, analyze what data is required and ensure:
1. All entities referenced in the task exist in the database
2. All relationships needed to complete the task are properly established
3. Query results will return meaningful, non-empty data
4. Edge cases and variations are covered for robust testing


Hard Requirements:
1. You must strictly follow the provided database schema - use EXACT table names, column names, and valid column counts. Do NOT invent new tables or columns.
2. Generate INSERT statements for ALL tables necessary to support the tasks, but ONLY for tables that exist in the schema.
3. Ensure data integrity: respect foreign key relationships and constraints.
4. Follow SQLite syntax for INSERT statements.
5. Insert data in the correct order to satisfy foreign key constraints.
6. If a users table exists, ALWAYS create user with id=1 as the first entry - this is the current authenticated user.
7. DO NOT include authentication fields like password_hash, token, session, even if they appear in the schema.
8. For user-owned data (orders, posts, etc.), create MOST data for user_id=1.
9. For each table in the output, provide a brief reasoning (<= 100 words) in its "reasoning" field explaining how that table's insert statements support the given user tasks and comply with the database schema.
10. Return ONLY valid JSON, no additional text. Do NOT include comments, ellipsis ("..."), or wrap the JSON in backticks or code fences.
11. Each element in "insert_statements" MUST be a single SQL INSERT statement string, ending with a single semicolon, and MUST NOT contain multiple statements or any JSON/text before or after the SQL.
12. All items in "insert_statements" MUST be plain strings (SQL statements only), not objects or nested JSON structures.


INSERT Statement Format (MANDATORY):
- Always explicitly list the column names you are inserting into in each INSERT statement
- Format: INSERT INTO table_name (col1, col2, ..., colN) VALUES (val1, val2, ..., valN);
- The number of listed columns MUST equal the number of values
- The parentheses after the table name MUST contain ONLY column names, never literal values
- For NULL values, use NULL (not empty string)
- For boolean columns, use 0 or 1
- For optional columns with defaults or autoincrement primary keys, either include them with a value or omit both the column and the corresponding value from the INSERT


Agent Task Coverage Requirements:
1. For SEARCH/FILTER tasks: create diverse data that matches AND does not match search criteria
2. For LIST/GET tasks: create multiple records (at least 5-10) to return meaningful results
3. For CREATE/POST tasks: ensure all referenced entities (users, categories, etc.) exist
4. For UPDATE/PATCH tasks: create existing records that can be modified
5. For DELETE tasks: create expendable records that can be safely deleted
6. For AGGREGATION tasks (count, sum, avg): create sufficient data volume for meaningful statistics
7. For RELATIONSHIP tasks: ensure all foreign key references are valid and queryable


Data Quality Requirements:
1. Use realistic values (real product names, proper email formats, realistic prices, etc.)
2. Create temporal diversity (records from different dates/times)
3. Include status variations (active/inactive, pending/completed, etc.)
4. Cover numeric ranges (low/medium/high prices, quantities, ratings)
5. Include text variations (short/long descriptions, different categories)
6. For timestamps, use ISO 8601 format: YYYY-MM-DD HH:MM:SS or datetime('now', '-N days')
7. Create enough data volume to support robust testing


Output format:
{{
    "tables": [
        {{
            "table_name": "users",
            "reasoning": "A brief explanation (<= 100 words) of how these insert statements support the given user tasks and strictly comply with the database schema.",
            "insert_statements": [
                "INSERT INTO users (id, username, email, full_name, created_at) VALUES (1, 'current_user', 'user@example.com', 'Current User', datetime('now', '-60 days'));",
                "INSERT INTO users (id, username, email, full_name, created_at) VALUES (2, 'other_user', 'other@example.com', 'Other User', datetime('now', '-30 days'));"
            ]
        }},
        {{
            "table_name": "products",
            "reasoning": "A brief explanation (<= 100 words) of how these insert statements support the given user tasks and strictly comply with the database schema.",
            "insert_statements": [
                "INSERT INTO products (name, description, price, user_id, created_at) VALUES ('Product A', 'Description A', 29.99, 1, datetime('now', '-20 days'));",
                "INSERT INTO products (name, description, price, user_id, created_at) VALUES ('Product B', 'Description B', 49.99, 1, datetime('now', '-15 days'));"
            ]
        }},
        ...
    ]
}}
"""


SCENARIO_CLASSIFICATION_SYSTEM_PROMPT = """You are an expert at analyzing websites, apps, and digital platforms to determine their suitability for API environment synthesis and simulation.

## What is "API Environment Synthesis"?
We want to create SIMULATED API servers that mimic real platforms. An agent will interact with these APIs to complete user tasks. The simulated environment must:
- Have a SQLite database with synthesized (fake but realistic) data
- Provide RESTful API endpoints for CRUD operations
- Support user workflows with SYNTHESIZED data (not real external data)

## KEY PRINCIPLE: Can the data be SYNTHESIZED?

### Data that CAN be synthesized (fake but realistic):
- Numeric values with reasonable ranges: temperatures, prices, quantities, ratings, coordinates
- Structured entities: users, products, orders, bookings, employees, transactions
- Status/state values: order status, booking status, task progress
- Short text: names, titles, descriptions, addresses, categories
- Dates and times: timestamps, schedules, deadlines
- Geographic data: cities, countries, latitude/longitude (static, not real-time)
- Tabular data: flight schedules, stock historical data, weather records

### Data that CANNOT be synthesized meaningfully:
- Long-form content: news articles, blog posts, encyclopedia entries, research papers
- Media content: videos, audio, images (the actual content, not metadata)
- AI inference: chatbot responses, recommendations based on ML models
- Search results: ranked results from a real search index
- Real-time external feeds: live sports scores, breaking news, live stock tickers

## Suitability Criteria:

### HIGH Suitability (score 8-10):
Core functionality involves CRUD operations on SYNTHESIZABLE data. Examples:
- E-commerce: Amazon (products, cart, orders, reviews)
- Task Management: Trello, Jira (tasks, boards, assignments)
- Banking: Chase (accounts, transactions, transfers)
- Booking: OpenTable, Airbnb (reservations, listings)
- Weather Services: Weather apps (locations, forecasts, alerts - all synthesizable numeric data!)
- Flight Booking: Airlines, Kayak (flights, prices, schedules - all synthesizable!)
- Stock Trading: Robinhood (portfolio, trades, historical prices - synthesizable!)
- CRM: Salesforce (leads, contacts, deals)
- HR Systems: Workday (employees, time-off, payroll)
- IoT/Smart Home: Thermostats, sensors (device status, readings - synthesizable numbers!)

Key characteristics:
- Users CREATE, UPDATE, DELETE their own data
- Data is STRUCTURED and can be realistically faked
- Clear entity relationships
- Even "external data" like weather/stocks can be simulated with fake but plausible values

### MEDIUM Suitability (score 5-7):
Has CRUD features, but some core features require non-synthesizable content. Examples:
- YouTube: Playlist/subscription management (simulatable), but video content is the core value
- Spotify: Playlist/library management (simulatable), but audio is the core value
- Reddit/Forums: Posts, comments, votes (partially simulatable with short content)
- Social Media: Profile, follows, messaging (simulatable), but feed content quality varies

Key characteristics:
- CRUD operations exist and can be simulated
- Some features depend on content that's hard to synthesize meaningfully
- Can focus on the management/organizational aspects

### LOW Suitability (score 1-4):
Core value is NON-SYNTHESIZABLE content or requires real external services. Examples:
- Wikipedia: Core value is article CONTENT that cannot be faked meaningfully
- News sites (BBC, CNN): Article content IS the product
- Search Engines: Google, Bing (requires real search index and ranking)
- AI/LLM Services: ChatGPT (requires model inference)
- Translation Services: Google Translate (requires real translation models)
- Image Recognition: Needs real ML models

Key characteristics:
- The "content" or "intelligence" IS the product
- Cannot be replaced with fake data without losing all value
- Requires real ML models or massive real content databases"""


SCENARIO_CLASSIFICATION_USER_PROMPT_TEMPLATE = """Analyze and classify the following website/app:

Name: {name}
Description: {description}

Provide a detailed classification:

1. **Categories**: What interaction pattern categories does this platform belong to? (select all that apply)
2. **Suitability Score**: 1-10 scale for API environment synthesis suitability
3. **Suitability Level**: "high", "medium", or "low"
4. **Reasoning**: Why is it suitable or not suitable?
5. **Simulatable Features**: What specific features CAN be simulated via CRUD APIs?
6. **Non-simulatable Features**: What features CANNOT be simulated?

Output format (JSON only, no markdown):
{{
    "categories": ["category1", "category2"],
    "suitability_score": 8,
    "suitability_level": "high",
    "reasoning": "Brief explanation",
    "simulatable_features": ["feature1", "feature2"],
    "non_simulatable_features": ["feature1", "feature2"]
}}"""


SCENARIO_CATEGORY_LIST = [
    "E-commerce/Marketplace",
    "Booking/Reservation",
    "Social/Community",
    "Task/Project Management",
    "Finance/Banking",
    "Subscription/Membership",
    "Inventory/Catalog",
    "Messaging/Communication",
    "Lists/Collections",
    "Scheduling/Calendar",
    "Forms/Surveys",
    "Settings/Configuration",
    "Healthcare/Medical",
    "Education/Learning",
    "Real Estate/Property",
    "HR/Recruiting",
    "Legal/Compliance",
    "Logistics/Shipping",
    "Food/Restaurant",
    "Entertainment/Gaming",
    "Fitness/Wellness",
    "Travel/Hospitality",
    "Automotive",
    "IoT/Smart Devices",
    "Developer Tools",
    "CRM/Sales",
    "Content Management",
    "Analytics/Reporting",
]


SCENARIO_GENERATION_SYSTEM_PROMPT = """You are an expert at identifying websites, apps, and digital platforms that are HIGHLY SUITABLE for API environment simulation and database-driven interactions.

## KEY PRINCIPLE: Can the data be SYNTHESIZED?

We need platforms where data can be FAKED but still be REALISTIC and USEFUL.

### Data that CAN be synthesized:
- Numbers: temperatures, prices, quantities, ratings, coordinates, stock prices
- Entities: users, products, orders, bookings, employees, devices, sensors
- Status values: order status, flight status, device state
- Short text: names, titles, addresses, categories, descriptions
- Timestamps: dates, schedules, deadlines, historical records
- Geographic: cities, airports, coordinates (static data)

### Data that CANNOT be synthesized meaningfully:
- Long articles: news content, blog posts, encyclopedia entries
- Media: actual video/audio content (not metadata)
- AI inference: chatbot responses, ML-based recommendations
- Real search: search engine results with ranking

## HIGH Suitability Examples (GENERATE THESE):

| Platform Type | Why Suitable | Key Entities (all synthesizable) |
|---------------|--------------|----------------------------------|
| E-commerce | CRUD on orders, reviews | products, orders, cart, reviews |
| Task Management | CRUD on tasks | tasks, boards, assignments |
| Banking/Fintech | CRUD on transactions | accounts, transactions, transfers |
| Booking/Reservation | CRUD on reservations | listings, bookings, availability |
| Weather Services | Query weather data | locations, forecasts, alerts, history |
| Flight/Travel | Search & book flights | flights, airports, bookings, prices |
| Stock Trading | Trade & portfolio | stocks, trades, portfolio, prices |
| IoT/Smart Home | Control devices | devices, sensors, readings, schedules |
| Fitness Tracking | Log workouts | workouts, exercises, metrics, goals |
| Healthcare | Manage appointments | patients, appointments, records, prescriptions |
| HR/Payroll | Manage employees | employees, timesheets, payroll |
| CRM/Sales | Manage leads | leads, deals, contacts, activities |
| Inventory | Manage stock | products, inventory, warehouses |
| Logistics | Track shipments | shipments, packages, routes, status |
| Restaurant/Food | Orders & menu | menu_items, orders, reservations |
| Property/Real Estate | Listings & viewings | properties, listings, viewings, offers |
| Education/LMS | Courses & grades | courses, enrollments, assignments, grades |
| Event Management | Events & tickets | events, tickets, attendees, venues |

## LOW Suitability (AVOID THESE):

| Platform Type | Why NOT Suitable | Problem |
|---------------|------------------|---------|
| News sites (BBC, CNN) | Article CONTENT is the product | Cannot synthesize meaningful articles |
| Wikipedia/Encyclopedia | Article CONTENT is the product | Cannot fake knowledge |
| Search Engines | Need real search index + ranking | Cannot simulate relevance |
| AI Assistants (ChatGPT) | Need real AI inference | Cannot fake AI responses |
| Translation (Google Translate) | Need real NLP models | Cannot fake translations |
| Content platforms focus | If CONTENT is core value | Metadata is not enough |

## MEDIUM Suitability (Focus on CRUD aspects):

| Platform | What CAN be simulated | What CANNOT |
|----------|----------------------|-------------|
| YouTube | Playlists, subs, comments, channel mgmt | Actual videos, recommendations |
| Spotify | Playlists, library, following | Actual audio, music discovery |
| Reddit | Posts (short), comments, votes | Long-form quality content |

## Categories to Cover (for diversity):

E-commerce, Booking/Reservation, Task/Project Management, Finance/Banking,
Weather/Environmental, Flight/Travel, Stock/Investment, IoT/Smart Home,
Fitness/Health Tracking, Healthcare/Medical, HR/Recruiting, CRM/Sales,
Inventory/Warehouse, Logistics/Shipping, Restaurant/Food, Real Estate,
Education/LMS, Event/Ticketing, Legal/Contracts, Subscription Management,
Customer Support, Forms/Surveys, Analytics Dashboards, Fleet Management,
Utility Management (electricity, water), Insurance, Pet Services, Automotive

## Your Task:
Generate NEW platforms that are HIGHLY SUITABLE - where data can be realistically SYNTHESIZED and users perform meaningful CRUD operations."""


SCENARIO_GENERATION_USER_PROMPT_TEMPLATE = """Here are {num_examples} examples of suitable website/app scenarios:

{examples}

---

Now generate {num_to_generate} NEW and DIVERSE website/app/platform scenarios that are DIFFERENT from all examples above.

Requirements:
1. Each scenario must be suitable for API environment synthesis (CRUD operations, database interactions)
2. Cover DIFFERENT interaction patterns and industries than the examples
3. Include a mix of: well-known platforms, niche services, B2B tools, mobile apps, and specialized tools
4. The description should emphasize: what entities users can manage, what operations are available, what workflows exist
5. Avoid content-heavy sites, real-time data sites, search engines, AI inference services
6. Each description should be 150-300 words, focusing on actionable features

Output format (JSON array, no markdown fences):
[
    {{"name": "Platform Name", "description": "Description focusing on CRUD operations, entities, and workflows..."}},
    ...
]

Generate exactly {num_to_generate} new scenarios:"""


SCENARIO_FOCUSED_GENERATION_USER_PROMPT_TEMPLATE = """Here are {num_examples} examples of suitable website/app scenarios:

{examples}

---

## IMPORTANT: Focus on generating scenarios in these specific categories:
{focus_categories}

Generate {num_to_generate} NEW and DIVERSE website/app/platform scenarios, with emphasis on the above categories.

Requirements:
1. Each scenario must be suitable for API environment synthesis (CRUD operations, database interactions)
2. Prioritize the focus categories above, but ensure variety within them
3. Include DIFFERENT sub-niches within each category (e.g., for E-commerce: fashion, electronics, B2B wholesale, auction, rental marketplace)
4. The description should emphasize: what entities users can manage, what operations are available, what workflows exist
5. Avoid content-heavy sites, real-time data sites, search engines, AI inference services
6. Each description should be 150-300 words, focusing on actionable features
7. Think of REAL platforms that exist, or realistic platforms that could exist

Output format (JSON array, no markdown fences):
[
    {{"name": "Platform Name", "description": "Description focusing on CRUD operations, entities, and workflows..."}},
    ...
]

Generate exactly {num_to_generate} new scenarios:"""


SCENARIO_FOCUS_CATEGORIES = [
    ["E-commerce/Marketplace", "Retail", "Auction", "Rental marketplace"],
    ["Booking/Reservation", "Travel", "Hospitality", "Appointments"],
    ["Social/Community", "Forums", "Professional networks", "Dating"],
    ["Task/Project Management", "Collaboration", "Workflow automation"],
    ["Finance/Banking", "Investment", "Cryptocurrency", "Payments"],
    ["Subscription/Membership", "Loyalty programs", "Digital content"],
    ["Inventory/Catalog", "Warehouse management", "Asset tracking"],
    ["Messaging/Communication", "Customer support", "Ticketing systems"],
    ["Lists/Collections", "Bookmarks", "Curation", "Wishlists"],
    ["Scheduling/Calendar", "Event management", "Resource booking"],
    ["Forms/Surveys", "Feedback", "Applications", "Registrations"],
    ["Healthcare/Medical", "Telehealth", "Patient portals", "Pharmacy"],
    ["Education/Learning", "LMS", "Tutoring", "Course platforms"],
    ["Real Estate/Property", "Rentals", "Property management"],
    ["HR/Recruiting", "Job boards", "Applicant tracking", "Payroll"],
    ["Legal/Compliance", "Contract management", "Document signing"],
    ["Logistics/Shipping", "Fleet management", "Delivery tracking"],
    ["Food/Restaurant", "Ordering", "Menu management", "Kitchen operations"],
    ["Entertainment/Gaming", "Streaming", "Game platforms", "Virtual goods"],
    ["IoT/Smart devices", "Home automation", "Device management"],
    ["Fitness/Wellness", "Workout tracking", "Gym management", "Nutrition"],
    ["Pet services", "Veterinary", "Pet sitting", "Pet supplies"],
    ["Automotive", "Car dealership", "Service scheduling", "Parts inventory"],
    ["Non-profit/Charity", "Donations", "Volunteer management", "Fundraising"],
]


SCENARIO_DIVERSITY_CHECK_SYSTEM_PROMPT = """You are a diversity checker for website/app scenarios. Your job is to identify which newly generated scenarios are too similar to existing ones or to each other.

Two scenarios are considered TOO SIMILAR if:
1. They serve the same specific niche (e.g., two restaurant reservation systems)
2. They have nearly identical core functionality (e.g., two generic project management tools)
3. They are direct competitors with no meaningful differentiation mentioned

Two scenarios are considered DIFFERENT ENOUGH if:
1. They serve different industries or user segments
2. They have distinct primary features or workflows
3. They target different scales (consumer vs enterprise, individual vs team)"""


SCENARIO_DIVERSITY_CHECK_USER_PROMPT_TEMPLATE = """## Existing Scenarios (these are already in our dataset):
{existing_scenarios}

## New Candidates to Check:
{new_candidates}

---

For each new candidate, determine if it is:
- "keep": Sufficiently different from all existing scenarios
- "reject": Too similar to an existing scenario (specify which one)

Also check for duplicates among the new candidates themselves.

Output format (JSON object, no markdown fences):
{{
    "decisions": [
        {{"index": 0, "name": "Candidate Name", "decision": "keep" or "reject", "reason": "brief reason", "similar_to": "name of similar scenario if rejected, else null"}},
        ...
    ],
    "summary": {{
        "total": N,
        "kept": M,
        "rejected": K
    }}
}}"""


SCENARIO_CATEGORY_DIVERSITY_PROMPT = """Analyze the following list of website/app scenarios and identify which interaction pattern categories are UNDERREPRESENTED.

Current scenarios:
{scenarios}

Interaction pattern categories:
1. E-commerce/Marketplace
2. Booking/Reservation
3. Social/Community
4. Task/Project Management
5. Finance/Banking
6. Subscription/Membership
7. Inventory/Catalog
8. Messaging/Communication
9. Lists/Collections
10. Scheduling/Calendar
11. Forms/Surveys
12. Settings/Configuration
13. Healthcare/Medical
14. Education/Learning
15. Real Estate/Property
16. HR/Recruiting
17. Legal/Compliance
18. Logistics/Shipping
19. Food/Restaurant
20. Entertainment/Gaming

Output format (JSON, no markdown fences):
{{
    "category_counts": {{"category_name": count, ...}},
    "underrepresented": ["category1", "category2", ...],
    "suggestions": ["Generate more X type scenarios", ...]
}}"""




VERIFIER_SQL_GENERATION_SYSTEM_PROMPT = """You are an expert Python and SQL developer. Your job is to generate Python code that uses SQLite queries to verify if a user task was completed successfully on a simplified API server. You only output UTF-8 encoded string, avoid any emoji or special characters. You only output English text."""

VERIFIER_SQL_GENERATION_USER_PROMPT = """You need to generate a Python function with sqlite3 queries to collect useful information from the given databases to verify if a user task was completed. You are now provided with the environment name of the simplified API server, the user task to verify, the database schema, and the initial database state to help you generate the Python function.

Simplified API Server Name:
{scenario_name}

User Task to Verify:
{user_task}

Database Dump (Initial State, before the agent takes any action):
{db_dump}
--------------------------------

Requirements:
1. Generate a complete Python function that takes initial_db_path and final_db_path as parameters.
2. The function should connect to the SQLite database and execute queries to return useful information to assist another LLM to judge the task completion.
3. You can use complex sqlite3 query combinations and Python logic. This function must return a dictionary containing the useful information for judging the task completion.
4. The python function and the queries should follow the database dump, do not invent any new tables or columns.
5. Use sqlite3 library, and import any other libraries you need
6. You will be provided with two database states: the initial database state and the final database state after the agent has executed the task. You can use the initial database state to help compare with the final database state.
7. You must NOT modify the databases in any way, you can only read the databases.
8. Ensure the python function can return a dictionary that can be json serialized. For example, do not use tuple as the key of the dictionary, do not use datetime and other non-json serializable objects. You can only use string, int, float, bool, list, and dict.
9. You must output a dictionary including:
    - reasoning: a concise explanation of why the function could be used to help verify the task is successful or failed. Also include the guidance for another LLM to judge based on the function's execution results.
    - python_code: the complete Python function code as a string that takes initial_db_path and final_db_path as parameters
    - function_name: the name of the function
    - success_criteria: a description of the expected returned results from the function that indicates the task was successful
    - failure_criteria: a description of the expected returned results from the function that indicates the task was failed

Example structure:
```python
def verify_task(initial_db_path: str, final_db_path: str) -> dict:
    import sqlite3
    import any other libraries the function needs

    conn_initial = sqlite3.connect(initial_db_path)
    conn_final = sqlite3.connect(final_db_path)

    # complex logic here
    # ... 
    
    conn_initial.close()
    conn_final.close()
    
    return {{
        you can return any valuable data or information here, but it should be able to be used to help another LLM determine if the task was completed successfully or failed. Ensure this dictionary can be json serialized, do not include any non-json serializable objects such as datetime, bytes, etc.
    }}
```

Output format (must be valid JSON, no markdown fences):
{{
    "reasoning": "Concise explanation of why the function could be used to verify the task is successful or failed",
    "python_code": "complete Python function code as a string that takes initial_db_path and final_db_path as parameters",
    "function_name": "verify_task",
    "success_criteria": "Description of the expected returned results from the function that indicates the task was successful",
    "failure_criteria": "Description of the expected returned results from the function that indicates the task was failed"
}}
"""


CODE_VERIFICATION_SYSTEM_PROMPT = """You are an expert Python developer specializing in database verification and test automation. Your job is to generate Python verification code that can definitively determine if an AI agent successfully completed a given task.

You only output UTF-8 encoded string, avoid any emoji or special characters. You only output English text.

CRITICAL REQUIREMENTS:
1. The verification code must be 100% deterministic and rule-based. If you cannot determine the results, just return {"result": "others"}.
2. The code must return {"result": "complete"} ONLY when you are 100% certain the task was completed
3. In ALL other cases (uncertainty, partial completion, errors), return {"result": "others"}
4. The code should handle edge cases gracefully without exceptions
5. Use defensive programming - assume inputs may be malformed or missing"""

CODE_VERIFICATION_USER_PROMPT = """You need to generate a Python verification function that DIRECTLY determines whether an AI agent successfully completed a task. This function will be used in Reinforcement Learning where NO additional LLM judgment is available - the code output is the FINAL verdict.

## Environment Information

Scenario Name: {scenario}

User Task to Verify:
{task}

Database Dump (Initial State, before the agent takes any action):
{db_dump}

## Function Signature

```python
def verify_task_completion(initial_db_path: str, final_db_path: str, final_answer: str | None = None) -> dict:
    \"\"\"
    Verify if the agent successfully completed the task.
    
    Args:
        initial_db_path: Path to the database BEFORE agent execution
        final_db_path: Path to the database AFTER agent execution  
        final_answer: The agent's final text response/summary (may be None)
    
    Returns:
        {{"result": "complete"}} if 100% certain task was completed successfully
        {{"result": "others"}} for any other case (incomplete, uncertain, error, partial)
    \"\"\"
```

## Verification Strategy Guidelines

### For Query/Read Tasks (e.g., "find...", "list...", "get...", "what is..."):
- The final_answer should contain the requested information
- Use regex patterns to extract and validate key information from final_answer
- Cross-reference extracted data with database content to verify accuracy
- Check if the answer format matches what was requested

### For Modification Tasks (e.g., "add...", "create...", "update...", "delete..."):
- Compare initial_db vs final_db to detect the expected changes
- Verify the specific records were created/modified/deleted as requested
- Check that the modifications match the task requirements exactly
- Validate data integrity (correct values, relationships, etc.)

### For Combined Tasks (query + action):
- Verify BOTH the database changes AND the final_answer content
- Ensure all sub-tasks are completed

## Code Requirements

1. Import all necessary libraries at the top of the function
2. Use sqlite3 for database operations
3. Use re module for regex pattern matching on final_answer
4. Handle None/empty final_answer gracefully
5. Use try-except ONLY for database connection errors, NOT for logic
6. Make the code robust to handle edge cases
7. Return {{"result": "complete"}} ONLY when ALL conditions are met with 100% certainty
8. Return {{"result": "others"}} in ANY other situation

## Output Format

You must output a valid JSON object (no markdown fences):
{{
    "reasoning": "Explain what checks are needed to verify this specific task",
    "verification_strategy": "query_based" | "modification_based" | "combined",
    "key_checks": ["list of specific checks the code will perform"],
    "python_code": "the complete Python function code as a string",
    "function_name": "verify_task_completion"
}}

## Example Code Structure

```python
def verify_task_completion(initial_db_path: str, final_db_path: str, final_answer: str | None = None) -> dict:
    import sqlite3
    import re
    
    # Helper to safely query database
    def safe_query(db_path: str, query: str, params: tuple = ()) -> list:
        try:
            conn = sqlite3.connect(f"file:{{db_path}}?mode=ro", uri=True)
            cursor = conn.cursor()
            cursor.execute(query, params)
            results = cursor.fetchall()
            conn.close()
            return results
        except Exception:
            return []
    
    # For query tasks: validate final_answer contains expected info
    # For modification tasks: compare database states
    # For combined: do both
    
    # Example check for modification task:
    # initial_count = len(safe_query(initial_db_path, "SELECT * FROM orders WHERE user_id = ?", (1,)))
    # final_count = len(safe_query(final_db_path, "SELECT * FROM orders WHERE user_id = ?", (1,)))
    # if final_count > initial_count:
    #     return {{"result": "complete"}}
    
    # Example check for query task:
    # if final_answer and re.search(r"price.*\\$[\\d.]+", final_answer, re.IGNORECASE):
    #     return {{"result": "complete"}}
    
    return {{"result": "others"}}
```

Now generate the verification code for the given task."""