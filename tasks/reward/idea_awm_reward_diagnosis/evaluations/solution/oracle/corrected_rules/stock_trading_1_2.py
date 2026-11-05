def verify_task_completion(initial_db_path: str, final_db_path: str, final_answer: str | None = None) -> dict:
    import sqlite3

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

    # Required symbols for the watchlist
    required_symbols = {'MSFT', 'NVDA', 'AMD', 'AVGO', 'META'}

    watchlist_query = "SELECT id FROM watchlists WHERE user_id = 1 AND name = 'Tech Growth 2025'"

    # Fix: the task is to CREATE a NEW watchlist. The original accepted the
    # identically named watchlist that already existed in the initial state
    # (with exactly the requested symbols), so a session that took no action
    # was rewarded. Only watchlists that are new relative to the initial
    # state can satisfy the request.
    initial_ids = {row[0] for row in safe_query(initial_db_path, watchlist_query)}
    final_ids = [row[0] for row in safe_query(final_db_path, watchlist_query)]

    if not final_ids:
        return {"result": "others"}

    # Get all securities in each watchlist with their symbols
    items_query = """
        SELECT s.symbol
        FROM watchlist_items wi
        JOIN securities s ON wi.security_id = s.id
        WHERE wi.watchlist_id = ?
    """

    for watchlist_id in final_ids:
        if watchlist_id in initial_ids:
            continue

        items_result = safe_query(final_db_path, items_query, (watchlist_id,))

        if not items_result:
            continue

        # Extract symbols from results
        watchlist_symbols = {row[0] for row in items_result}

        # Check if all required symbols are present and no extra symbols
        if watchlist_symbols == required_symbols:
            return {"result": "complete"}

    return {"result": "others"}
