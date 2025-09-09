query_std = f"""
        WITH counts AS (
        SELECT DISTINCT 'westeurope_spire_platform_prd.application_rdr_standard.std_us58_pd_contract' AS table_name, ctt_delivering_entity AS delivering_entity, max(ctt_reporting_date) AS max_reporting_date, COUNT(*) AS portfolio_count 
        FROM westeurope_spire_platform_prd.application_rdr_standard.std_us58_pd_contract
        GROUP BY ctt_delivering_entity, ctt_reporting_date
        UNION ALL
        SELECT DISTINCT 'westeurope_spire_platform_prd.application_rdr_standard.std_benl44_pd_contract' AS table_name, ctt_delivering_entity AS delivering_entity, max(ctt_reporting_date) AS max_reporting_date, COUNT(*) AS portfolio_count 
        FROM westeurope_spire_platform_prd.application_rdr_standard.std_benl44_pd_contract
        GROUP BY ctt_delivering_entity, ctt_reporting_date
        UNION ALL
        SELECT DISTINCT 'westeurope_spire_platform_prd.application_rdr_standard.std_at46_pd_contract' AS table_name, ctt_delivering_entity AS delivering_entity, max(ctt_reporting_date) AS max_reporting_date, COUNT(*) AS portfolio_count 
        FROM westeurope_spire_platform_prd.application_rdr_standard.std_at46_pd_contract
        GROUP BY ctt_delivering_entity, ctt_reporting_date
        UNION ALL
        SELECT DISTINCT 'westeurope_spire_platform_prd.application_rdr_standard.std_in36_pd_contract' AS table_name, ctt_delivering_entity AS delivering_entity, max(ctt_reporting_date) AS max_reporting_date, COUNT(*) AS portfolio_count 
        FROM westeurope_spire_platform_prd.application_rdr_standard.std_in36_pd_contract
        GROUP BY ctt_delivering_entity, ctt_reporting_date
        UNION ALL
        SELECT DISTINCT 'westeurope_spire_platform_prd.application_rdr_standard.std_ca61_pd_contract' AS table_name, ctt_delivering_entity AS delivering_entity, max(ctt_reporting_date) AS max_reporting_date, COUNT(*) AS portfolio_count 
        FROM westeurope_spire_platform_prd.application_rdr_standard.std_ca61_pd_contract
        GROUP BY ctt_delivering_entity, ctt_reporting_date
        UNION ALL
        SELECT DISTINCT 'westeurope_spire_platform_prd.application_rdr_standard.std_au26_pd_contract' AS table_name, ctt_delivering_entity AS delivering_entity, max(ctt_reporting_date) AS max_reporting_date, COUNT(*) AS portfolio_count 
        FROM westeurope_spire_platform_prd.application_rdr_standard.std_au26_pd_contract
        GROUP BY ctt_delivering_entity, ctt_reporting_date
    ),
    total_count AS (
        SELECT SUM(portfolio_count) AS total FROM counts
    ),
    percentages AS (
        SELECT 
            table_name,
            delivering_entity,
            max_reporting_date, 
            portfolio_count, 
            ROUND((portfolio_count * 100.0 / total_count.total), 2) AS percentage_of_grand_total,
            ROUND((portfolio_count * 5500.0 / total_count.total)) AS rows_to_select
        FROM counts, total_count
    )
    SELECT 
        table_name,
        delivering_entity,
        max_reporting_date,
        portfolio_count, 
        percentage_of_grand_total,
        rows_to_select,
        concat(ROUND((rows_to_select * 100.0 / portfolio_count),2),"% / 100.00%") AS portfolio_percentage
    FROM percentages
    ORDER BY rows_to_select DESC
    """

query_hist = """
WITH counts AS (
        SELECT 
            fact_pd_contract.ctt_delivering_entity delivering_entity,
            COUNT(*) AS portfolio_count
        FROM westeurope_spire_platform_prd.application_rdr_historymodel.fact_pd_contract
        WHERE fact_pd_contract.ctt_delivering_entity IN ('58', '44', '46', '36', '61', '26')
        GROUP BY fact_pd_contract.ctt_delivering_entity
    ),
    total_count AS (
        SELECT SUM(portfolio_count) AS total FROM counts
    ),
    percentages AS (
        SELECT
            delivering_entity,
            portfolio_count, 
            ROUND((portfolio_count * 100.0 / total_count.total), 2) AS percentage_of_grand_total,
            ROUND((portfolio_count * 500000.0 / total_count.total)) AS rows_to_select
        FROM counts, total_count
    )
    SELECT
        delivering_entity,
        portfolio_count, 
        percentage_of_grand_total,
        rows_to_select,
        concat(ROUND((rows_to_select * 100.0 / portfolio_count),2),"% / 100.00%") AS portfolio_percentage
    FROM percentages
    ORDER BY rows_to_select DESC
"""