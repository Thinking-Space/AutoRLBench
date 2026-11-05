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

    # Check if the thread exists with correct title, community, and category
    thread_query = """
        SELECT t.id, t.status, t.title
        FROM threads t
        WHERE t.community_id = 1
        AND t.category_id = 1
        AND t.title = 'Saturday Park Cleanup Volunteers Needed'
    """

    # Fix: the task is to CREATE a NEW thread. The original accepted the
    # identically titled thread that already existed in the initial state
    # (with compliant tags and initial post), so a session that took no action
    # was rewarded. Only threads that are new relative to the initial state
    # can satisfy the request.
    initial_ids = {row[0] for row in safe_query(initial_db_path, thread_query)}
    final_threads = safe_query(final_db_path, thread_query)

    if not final_threads or len(final_threads) == 0:
        return {"result": "others"}

    for thread_id, thread_status, thread_title in final_threads:
        if thread_id in initial_ids:
            continue

        # Verify thread status is 'open'
        if thread_status != 'open':
            continue

        # Verify the thread has the correct tags
        tags_query = """
            SELECT tg.name
            FROM thread_tags tt
            JOIN tags tg ON tt.tag_id = tg.id
            WHERE tt.thread_id = ?
            ORDER BY tg.name
        """

        tags = safe_query(final_db_path, tags_query, (thread_id,))
        tag_names = [tag[0] for tag in tags]

        required_tags = ['cleanup', 'park', 'volunteer']
        if sorted(tag_names) != sorted(required_tags):
            continue

        # Verify the initial post exists and contains correct content
        post_query = """
            SELECT p.content
            FROM posts p
            WHERE p.thread_id = ?
            AND p.position = 1
        """

        posts = safe_query(final_db_path, post_query, (thread_id,))

        if not posts or len(posts) == 0:
            continue

        post_content = posts[0][0].lower()

        # Check for key content elements in the post
        # Must mention: Saturday, time (9-11am or 9–11am), Brookside Park, cleanup/trash pickup, and request for replies/volunteers
        has_saturday = 'saturday' in post_content
        has_time = bool(re.search(r'9[\s]*[-–—][\s]*11[\s]*a\.?m\.?', post_content, re.IGNORECASE))
        has_park = 'brookside park' in post_content
        has_cleanup = 'trash' in post_content or 'cleanup' in post_content or 'pick' in post_content
        has_request = 'reply' in post_content or 'interest' in post_content or 'join' in post_content or 'volunteer' in post_content

        if not (has_saturday and has_time and has_park and has_cleanup and has_request):
            continue

        # All checks passed
        return {"result": "complete"}

    return {"result": "others"}
