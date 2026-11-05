def verify_task_completion(initial_db_path: str, final_db_path: str, final_answer: str | None = None) -> dict:
    import sqlite3
    import re

    def safe_query(db_path: str, query: str, params: tuple = ()) -> list:
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            cursor = conn.cursor()
            cursor.execute(query, params)
            results = cursor.fetchall()
            conn.close()
            return results
        except Exception:
            return []

    # Check if final_answer exists and has content
    if not final_answer or len(final_answer.strip()) < 10:
        return {"result": "others"}

    # Find the campaign and team IDs
    campaign_query = """
        SELECT id FROM campaigns
        WHERE name = '2025 Summer Meal Drive'
    """
    campaign_results = safe_query(final_db_path, campaign_query)
    if not campaign_results:
        return {"result": "others"}
    campaign_id = campaign_results[0][0]

    team_query = """
        SELECT id FROM teams
        WHERE name = 'Downtown Champions' AND campaign_id = ?
    """
    team_results = safe_query(final_db_path, team_query, (campaign_id,))
    if not team_results:
        return {"result": "others"}
    team_id = team_results[0][0]

    # Get leaderboard data from database
    leaderboard_query = """
        SELECT
            tm.name,
            tm.id,
            COALESCE(SUM(CASE WHEN d.status = 'succeeded' THEN d.amount ELSE 0 END), 0) as total_raised,
            COUNT(CASE WHEN d.status = 'succeeded' THEN d.id ELSE NULL END) as num_donations
        FROM team_members tm
        LEFT JOIN donations d ON d.team_member_id = tm.id AND d.team_id = ?
        WHERE tm.team_id = ?
        GROUP BY tm.id, tm.name
        ORDER BY total_raised DESC
        LIMIT 10
    """
    leaderboard_results = safe_query(final_db_path, leaderboard_query, (team_id, team_id))

    if not leaderboard_results:
        return {"result": "others"}

    # Build expected leaderboard with rankings
    expected_members = []
    rank = 1
    for idx, (name, member_id, total_raised, num_donations) in enumerate(leaderboard_results):
        expected_members.append({
            'name': name,
            'total_raised': float(total_raised),
            'num_donations': int(num_donations),
            'rank': rank
        })
        # Update rank for next iteration (handle ties by checking if next total is different)
        if idx + 1 < len(leaderboard_results) and leaderboard_results[idx + 1][2] < total_raised:
            rank = idx + 2

    # Verify each member appears in final_answer with their data
    members_found = 0
    final_answer_lower = final_answer.lower()
    expected_names_lower = [member['name'].lower() for member in expected_members]

    for member in expected_members:
        name_lower = member['name'].lower()

        # Check if member name appears
        position = final_answer_lower.find(name_lower)
        if position < 0:
            continue

        # The member's entry runs from their name to the next expected
        # member's name (or the end of the answer)
        segment_start = position + len(name_lower)
        segment_end = len(final_answer_lower)
        for other_lower in expected_names_lower:
            if other_lower == name_lower:
                continue
            other_position = final_answer_lower.find(other_lower, segment_start)
            if other_position != -1:
                segment_end = min(segment_end, other_position)
        segment = final_answer_lower[segment_start:segment_end]

        # Look for their total raised amount (allow various formats: 500, 500.0, $500, etc.)
        total_str = str(member['total_raised'])
        if '.' in total_str:
            total_int = total_str.split('.')[0]
        else:
            total_int = total_str

        # Check for amount in various formats within the member's entry
        amount_patterns = [
            rf"\${total_int}(?:\.0+)?\b",
            rf"\b{total_int}(?:\.0+)?\s*(?:dollars?|usd)?\b",
            rf"\b{member['total_raised']}\b"
        ]
        amount_found = any(re.search(pattern, segment, re.IGNORECASE) for pattern in amount_patterns)

        # Fix: the task asks for each member's number of donations. The
        # original computed a per-member donation-count presence flag but did
        # not use it, so answers that omitted every donation count were
        # rewarded. The count must now appear as a labeled number within the
        # member's entry and match the computed value.
        count_match = re.search(r'(\d+)\s+donations?', segment) or re.search(
            r'donations?\s*[:=]?\s*(\d+)', segment
        )
        count_found = bool(count_match) and int(count_match.group(1)) == member['num_donations']

        # Member is considered found only if name, amount, and donation count
        # are all present and correct
        if amount_found and count_found:
            members_found += 1

    # We need to find at least the majority of members (at least 80% of expected)
    min_required = max(1, int(len(expected_members) * 0.8))

    if members_found < min_required:
        return {"result": "others"}

    # Check for descending order indicator (top members should appear before lower ones)
    # Verify the top member appears in the answer
    if expected_members:
        top_member_name = expected_members[0]['name'].lower()
        if top_member_name not in final_answer_lower:
            return {"result": "others"}

    # Check for key terms indicating this is a leaderboard
    leaderboard_terms = ['rank', 'total', 'raised', 'donation', 'amount', 'member']
    terms_found = sum(1 for term in leaderboard_terms if term in final_answer_lower)

    if terms_found < 3:
        return {"result": "others"}

    # All checks passed
    return {"result": "complete"}
