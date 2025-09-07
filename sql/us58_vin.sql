select
layer_name,
concat('a',substring(md5(ctt_cms_contract_number),3,8),'xy',substring(md5(ctt_cms_contract_number),4,9),'z') ctt_cms_contract_number,
CASE
WHEN warning_description = 'Warning Prio 2/VIN should not contain I, O and Q' AND regexp_substr(ctt_vin, '[^A-HJ-NPR-Z0-9]') IS NOT NULL THEN concat('a',substring(md5(ctt_vin),3,8),regexp_substr(ctt_vin, '[^A-HJ-NPR-Z0-9]'),substring(md5(ctt_vin),4,9),'z')
ELSE concat('a',substring(md5(ctt_vin),3,8),'xy',substring(md5(ctt_vin),4,9),'z')
end ctt_vin,
ctt_asset_type_segment,
entity_name,
reporting_date,
case
when warning_description in ("Warning Prio 2/Duplicate of VIN", "Warning Prio 1/Mandatory field missing or invalid - VIN", "Warning Prio 2/VIN should not contain special characters", 
  "Warning Prio 2/VIN should not contain I, O and Q", "Warning Prio 2/No matches found within the Mapis table lookup for the VIN") then warning_description
else "No anomaly detected for VIN"
end as anomaly_description,
case
when warning_description in ("Warning Prio 2/Duplicate of VIN", "Warning Prio 1/Mandatory field missing or invalid - VIN", "Warning Prio 2/VIN should not contain special characters", 
  "Warning Prio 2/VIN should not contain I, O and Q", "Warning Prio 2/No matches found within the Mapis table lookup for the VIN") then "1"
else "0"
end as anomaly_flag
from
(
WITH
std_layer as
(
  select
  distinct
  'Standard' layer_name, 
  std.ctt_cms_contract_number,
  std.ctt_vin,
  std.ctt_asset_type_segment,
  std.ctt_delivering_entity entity_name,
  std.ctt_reporting_date reporting_date,
  Description warning_description
  from westeurope_spire_platform_prd.application_rdr_standard.std_us58_pd_contract std --Apply python filter here
  left join westeurope_spire_platform_prd.application_rdr_metadata.dqlog dqlog
  on std.ctt_cms_contract_number = get_json_object(dqlog.Output,'$.ctt_cms_contract_number')
  AND dqlog.FwkEntityId LIKE '%us58_pd_contract' --Apply python filter here
  LIMIT 4219 --Apply python filter here
),
hist_layer as
(
  select
  distinct
  'History' layer_name,
  hist.ctt_cms_contract_number,
  hist.ctt_vin,
  hist.ctt_asset_type_segment,
  hist.ctt_delivering_entity entity_name,
  hist.ctt_reporting_date reporting_date,
  Description warning_description
  from 
  westeurope_spire_platform_prd.application_rdr_historymodel.fact_pd_contract hist
  left join westeurope_spire_platform_prd.application_rdr_metadata.dqlog dqlog
  on hist.ctt_cms_contract_number = get_json_object(Output,'$.ctt_cms_contract_number')
  AND hist.ctt_delivering_entity = "58" --Apply python filter here
  AND dqlog.FwkEntityId LIKE '%us58_pd_contract' --Apply python filter here
  LIMIT 353826 --Apply python filter here
)
SELECT *
from std_layer
union
SELECT *
from hist_layer
WHERE NOT EXISTS
(
    select 1
    from std_layer
    WHERE std_layer.ctt_cms_contract_number = hist_layer.ctt_cms_contract_number
)
)