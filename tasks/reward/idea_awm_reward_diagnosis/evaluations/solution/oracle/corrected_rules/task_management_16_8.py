def verify_task_completion(initial_db_path: str, final_db_path: str, final_answer: str | None = None) -> dict:
    import sqlite3
    import json
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

    dashboard_name = 'Marketing Campaign Overview'
    current_user = 1

    # Fix: the task asks the CURRENT USER to CREATE a dashboard with this name.
    # The original matched any dashboard by name, so a pre-existing dashboard
    # owned by another user satisfied the request even though the current
    # user's newly created dashboard was left empty. Only dashboards that are
    # new relative to the initial state and owned by the current user can
    # satisfy the request.
    initial_ids = {
        row[0]
        for row in safe_query(initial_db_path, "SELECT id FROM dashboards WHERE name = ?", (dashboard_name,))
    }
    final_rows = safe_query(
        final_db_path, "SELECT id, created_by FROM dashboards WHERE name = ?", (dashboard_name,)
    )
    candidate_ids = [
        row[0] for row in final_rows if row[0] not in initial_ids and row[1] == current_user
    ]

    if not candidate_ids:
        return {"result": "others"}

    for dashboard_id in candidate_ids:
        # Verify dashboard is linked to both required boards
        board_links_query = "SELECT board_id FROM dashboard_boards WHERE dashboard_id = ? ORDER BY board_id"
        board_links = safe_query(final_db_path, board_links_query, (dashboard_id,))

        linked_board_ids = [link[0] for link in board_links]
        if 4 not in linked_board_ids or 5 not in linked_board_ids:
            continue

        # Get all widgets for this dashboard
        widgets_query = "SELECT widget_type, title, settings_json FROM widgets WHERE dashboard_id = ?"
        widgets = safe_query(final_db_path, widgets_query, (dashboard_id,))

        if not widgets or len(widgets) < 3:
            continue

        # Track required widgets
        has_numbers_widget = False
        has_chart_widget = False
        has_timeline_widget = False

        for widget_type, title, settings_json_str in widgets:
            if not settings_json_str:
                continue

            try:
                settings = json.loads(settings_json_str)
            except:
                continue

            # Check numbers widget for Launched status count
            if widget_type == 'numbers':
                boards_in_settings = settings.get('boards', [])
                status_value = settings.get('status_value', '')

                if (4 in boards_in_settings and 5 in boards_in_settings and
                        status_value.lower() == 'launched'):
                    has_numbers_widget = True

            # Check chart widget for status grouping
            elif widget_type == 'chart':
                boards_in_settings = settings.get('boards', [])
                group_by = settings.get('group_by', '')

                if (4 in boards_in_settings and 5 in boards_in_settings and
                        group_by.lower() == 'status'):
                    has_chart_widget = True

            # Check timeline widget for date columns and active filter
            elif widget_type == 'timeline':
                boards_in_settings = settings.get('boards', [])
                start_date_cols = settings.get('start_date_columns', [])
                end_date_cols = settings.get('end_date_columns', [])
                filter_status = settings.get('filter_status', [])

                # Verify it includes both boards
                if not (4 in boards_in_settings and 5 in boards_in_settings):
                    continue

                # Verify date columns are configured (columns 15,18 for start, 16,19 for end)
                has_start_dates = len(start_date_cols) >= 2
                has_end_dates = len(end_date_cols) >= 2

                # Verify filter includes active statuses (Planned and/or Launched)
                has_active_filter = False
                if filter_status:
                    for status in filter_status:
                        status_lower = status.lower()
                        if status_lower in ['planned', 'launched']:
                            has_active_filter = True
                            break

                if has_start_dates and has_end_dates and has_active_filter:
                    has_timeline_widget = True

        # All three required widgets must be present and correctly configured
        if has_numbers_widget and has_chart_widget and has_timeline_widget:
            return {"result": "complete"}

    return {"result": "others"}
