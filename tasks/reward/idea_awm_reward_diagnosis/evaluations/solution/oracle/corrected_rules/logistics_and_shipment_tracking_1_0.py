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

    # Get merchant_id for 'acme_store'
    merchant_query = "SELECT id FROM merchants WHERE merchant_code = ?"
    merchant_result = safe_query(final_db_path, merchant_query, ('acme_store',))
    if not merchant_result:
        return {"result": "others"}
    merchant_id = merchant_result[0][0]

    # Get carrier_id for 'fastship'
    carrier_query = "SELECT id FROM carriers WHERE carrier_code = ?"
    carrier_result = safe_query(final_db_path, carrier_query, ('fastship',))
    if not carrier_result:
        return {"result": "others"}
    carrier_id = carrier_result[0][0]

    # Get service_level_id for 'standard_ground' under 'fastship'
    service_query = "SELECT id FROM service_levels WHERE carrier_id = ? AND service_code = ?"
    service_result = safe_query(final_db_path, service_query, (carrier_id, 'standard_ground'))
    if not service_result:
        return {"result": "others"}
    service_level_id = service_result[0][0]

    # Check for shipments with the specified order_reference and merchant
    shipment_query = """
        SELECT id, carrier_id, service_level_id, status, declared_value,
               length_cm, width_cm, height_cm, weight_kg
        FROM shipments
        WHERE merchant_id = ? AND order_reference = ?
    """
    final_shipments = safe_query(final_db_path, shipment_query, (merchant_id, 'ACME-ORD-1001'))

    if not final_shipments:
        return {"result": "others"}

    # Fix: the task is to CREATE a NEW shipment. The original accepted a
    # shipment that already existed in the initial state (it queried the
    # initial database and ignored the result), so a session that took no
    # action was rewarded. Only shipments that are new relative to the initial
    # state can satisfy the request.
    initial_ids = {
        row[0]
        for row in safe_query(initial_db_path, shipment_query, (merchant_id, 'ACME-ORD-1001'))
    }
    new_shipments = [row for row in final_shipments if row[0] not in initial_ids]

    for shipment in new_shipments:
        shipment_id = shipment[0]
        actual_carrier_id = shipment[1]
        actual_service_level_id = shipment[2]
        actual_status = shipment[3]
        actual_declared_value = shipment[4]
        actual_length = shipment[5]
        actual_width = shipment[6]
        actual_height = shipment[7]
        actual_weight = shipment[8]

        # Verify all shipment attributes match the requirements
        if actual_carrier_id != carrier_id:
            continue
        if actual_service_level_id != service_level_id:
            continue
        if actual_status != 'created':
            continue

        # Check declared_value with tolerance for floating point
        if actual_declared_value is None or abs(actual_declared_value - 79.99) > 0.01:
            continue

        # Check dimensions with tolerance
        if actual_length is None or abs(actual_length - 30.0) > 0.01:
            continue
        if actual_width is None or abs(actual_width - 20.0) > 0.01:
            continue
        if actual_height is None or abs(actual_height - 10.0) > 0.01:
            continue
        if actual_weight is None or abs(actual_weight - 2.5) > 0.01:
            continue

        # Check delivery address
        address_query = """
            SELECT address_line, city, region, postal_code, country, address_type
            FROM shipment_addresses
            WHERE shipment_id = ? AND address_type = 'delivery'
        """
        address_result = safe_query(final_db_path, address_query, (shipment_id,))

        if not address_result:
            continue

        address = address_result[0]
        if address[0] != '123 Market St':
            continue
        if address[1] != 'San Francisco':
            continue
        if address[2] != 'CA':
            continue
        if address[3] != '94103':
            continue
        if address[4] != 'US':
            continue

        # A new shipment with every requested attribute exists in the final state
        return {"result": "complete"}

    return {"result": "others"}
