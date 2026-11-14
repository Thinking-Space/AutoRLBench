def verify_task_completion(initial_db_path: str, final_db_path: str, final_answer: str | None = None) -> dict:
    import sqlite3
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

    # Fix: the task asks to CREATE a new firmware version (version_code
    # '2.3.5', release_channel 'stable', compatibility limited to
    # vm-COOLER-200 / vm-COOLER-300) and then SCHEDULE a new update job for
    # it. The original accepted the seed's pre-existing firmware row and a
    # job that already existed in the initial state, so a session that took
    # no action was rewarded. Only firmware versions and jobs that are new
    # relative to the initial state can satisfy the request.
    initial_fw_ids = {
        row[0] for row in safe_query(initial_db_path, "SELECT id FROM firmware_versions")
    }

    # Step 1: Find firmware_version with version_code='2.3.5' and release_channel='stable'
    fw_query = "SELECT id FROM firmware_versions WHERE version_code = ? AND release_channel = ?"
    fw_results = [
        (row[0],)
        for row in safe_query(final_db_path, fw_query, ('2.3.5', 'stable'))
        if row[0] not in initial_fw_ids
    ]

    if not fw_results:
        return {"result": "others"}

    fw_version_id = fw_results[0][0]

    # Step 2: Check compatibility entries for this firmware_version
    compat_query = "SELECT device_model_id FROM firmware_version_compatibility WHERE firmware_version_id = ? ORDER BY device_model_id"
    compat_results = safe_query(final_db_path, compat_query, (fw_version_id,))

    if not compat_results:
        return {"result": "others"}

    compatible_models = sorted([row[0] for row in compat_results])
    expected_models = sorted(['vm-COOLER-200', 'vm-COOLER-300'])

    # Check if compatibility includes at least the required models
    if not all(model in compatible_models for model in expected_models):
        return {"result": "others"}

    # Step 3: Find firmware_update_job with the specified parameters
    # The job should target site_type='factory', device_status='active'
    # Start time should be '2026-01-02T01:00:00Z' (stored as '2026-01-02 01:00:00')
    # End time should be 3 hours later: '2026-01-02 04:00:00'
    job_query = """
        SELECT id, firmware_version_id, release_channel, target_site_type,
               target_device_status, target_device_model_id,
               rollout_start_at, rollout_end_at
        FROM firmware_update_jobs
        WHERE rollout_start_at = ? AND target_site_type = ? AND target_device_status = ?
    """

    # Try different datetime formats
    start_time_formats = ['2026-01-02 01:00:00', '2026-01-02T01:00:00', '2026-01-02T01:00:00Z']
    job_results = []
    initial_job_ids = set()

    for start_format in start_time_formats:
        job_results = safe_query(final_db_path, job_query, (start_format, 'factory', 'active'))
        if job_results:
            initial_job_ids = {
                row[0]
                for row in safe_query(initial_db_path, job_query, (start_format, 'factory', 'active'))
            }
            break

    if not job_results:
        return {"result": "others"}

    # The update job must also be new relative to the initial state.
    for job_row in job_results:
        job_id, job_fw_id, job_channel, job_site_type, job_status, job_model_id, job_start, job_end = job_row

        if job_id in initial_job_ids:
            continue

        # Job must be linked to the firmware_version with id matching our fw_version_id
        if job_fw_id != fw_version_id:
            continue

        # Release channel should be 'stable'
        if job_channel != 'stable':
            continue

        # target_device_model_id should be NULL (to target all models of those
        # types at the site, per the firmware compatibility entries)

        # Check rollout window is 3 hours
        if not job_end:
            continue

        # Parse start and end times
        try:
            # SQLite stores as string, parse to compare
            start_dt = None
            end_dt = None

            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%SZ']:
                try:
                    start_dt = datetime.strptime(job_start, fmt)
                    break
                except ValueError:
                    continue

            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%SZ']:
                try:
                    end_dt = datetime.strptime(job_end, fmt)
                    break
                except ValueError:
                    continue

            if not start_dt or not end_dt:
                continue

            # Check if end is exactly 3 hours after start
            expected_end = start_dt + timedelta(hours=3)
            if end_dt != expected_end:
                continue

            # All checks passed
            return {"result": "complete"}

        except Exception:
            continue

    return {"result": "others"}
