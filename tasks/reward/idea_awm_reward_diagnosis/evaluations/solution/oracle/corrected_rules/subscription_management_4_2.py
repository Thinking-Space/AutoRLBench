def verify_task_completion(initial_db_path: str, final_db_path: str, final_answer: str | None = None) -> dict:
    import sqlite3
    import re
    from datetime import datetime, timedelta

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

    # Check if final_answer exists
    if not final_answer or not isinstance(final_answer, str) or len(final_answer.strip()) == 0:
        return {"result": "others"}

    # Get the current user's active subscription and max_simultaneous_devices
    subscription_query = """
        SELECT sp.max_simultaneous_devices, us.plan_id
        FROM user_subscriptions us
        JOIN subscription_plans sp ON us.plan_id = sp.id
        WHERE us.user_id = 1 AND us.status = 'active'
    """

    subscription_results = safe_query(final_db_path, subscription_query)

    if not subscription_results or len(subscription_results) == 0:
        return {"result": "others"}

    max_devices = subscription_results[0][0]

    if max_devices is None or not isinstance(max_devices, int) or max_devices < 0:
        return {"result": "others"}

    # Fix part 1 (window anchor): "the last 30 days" means the 30 days
    # preceding the current date. The original anchored the window to the most
    # recent stored play_session, which resurrects stale sessions into the
    # window and undercounts the remaining devices.
    now = datetime.now()
    window_start = (now - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
    now_str = now.strftime('%Y-%m-%d %H:%M:%S')

    device_query = """
        SELECT COUNT(DISTINCT device_type)
        FROM play_sessions
        WHERE user_id = 1
        AND start_time >= ? AND start_time <= ?
    """

    device_results = safe_query(final_db_path, device_query, (window_start, now_str))

    if not device_results or device_results[0][0] is None:
        distinct_devices = 0
    else:
        distinct_devices = device_results[0][0]

    # Calculate expected additional devices
    additional_devices = max_devices - distinct_devices

    # Ensure additional_devices is not negative (edge case)
    if additional_devices < 0:
        additional_devices = 0

    # Fix part 2 (conclusion check): every count the answer asserts as the
    # number of additional devices must equal the computed value. The original
    # searched the whole answer text for any occurrence of the number, and
    # additionally accepted any answer that merely mentioned the plan limit
    # and some other number, so answers whose stated conclusion was wrong
    # were rewarded.
    answer_lower = final_answer.lower()

    claims = set()
    for match in re.finditer(r'(\d+)\s+additional\s+devices?', answer_lower):
        claims.add(int(match.group(1)))
    for match in re.finditer(
        r'additional\s+devices?\s+(?:is|are|=|:)\s*(\d+)', answer_lower
    ):
        claims.add(int(match.group(1)))
    for match in re.finditer(
        r'(?:can|could)\s+(?:still\s+)?(?:use|add|connect|enable)\s+(\d+)\s+'
        r'(?:more\s+)?(?:additional\s+)?devices',
        answer_lower,
    ):
        claims.add(int(match.group(1)))

    if not claims:
        return {"result": "others"}

    if any(claim != additional_devices for claim in claims):
        return {"result": "others"}

    return {"result": "complete"}
