WITH counts AS (
        SELECT
            fact_pd_contract.ctt_analysis_entity analysis_entity,
            fact_pd_contract.ctt_delivering_entity delivering_entity,
            COUNT(*) AS portfolio_count
        FROM westeurope_spire_platform_prd.application_rdr_historymodel.fact_pd_contract
        WHERE fact_pd_contract.ctt_delivering_entity IN ('58', '46', '36', '61', '26')
        GROUP BY fact_pd_contract.ctt_analysis_entity, fact_pd_contract.ctt_delivering_entity
    ),
    total_count AS (
        SELECT SUM(portfolio_count) AS total FROM counts
    ),
    percentages AS (
        SELECT
            analysis_entity,
            delivering_entity,
            portfolio_count, 
            ROUND((portfolio_count * 100.0 / total_count.total), 2) AS percentage_of_grand_total,
            ROUND((portfolio_count * 500000.0 / total_count.total)) AS rows_to_select
        FROM counts, total_count
    )
    SELECT
        analysis_entity,
        delivering_entity,
        portfolio_count, 
        percentage_of_grand_total,
        rows_to_select,
        concat(ROUND((rows_to_select * 100.0 / portfolio_count),2),"% / 100.00%") AS portfolio_percentage
    FROM percentages
    ORDER BY
    CASE delivering_entity
        WHEN '58' THEN 1
        WHEN '46' THEN 2
        WHEN '36' THEN 3
        WHEN '61' THEN 4
        WHEN '26' THEN 5
        ELSE 999
    END