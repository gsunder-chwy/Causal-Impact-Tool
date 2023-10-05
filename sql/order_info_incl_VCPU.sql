DROP TABLE IF Exists core_locations;
CREATE LOCAL TEMP TABLE core_locations ON COMMIT PRESERVE ROWS AS 
SELECT
    location_key,
    TRIM(location_code) as location_code -- fulfillment center's name
FROM
    chewybi.locations
WHERE
    (
        fulfillment_active = 'true'
        --OR fulfillment_active IS NULL
    )
    AND (
        location_warehouse_type = 0
        OR location_warehouse_type IS NULL
    )
    AND location_active_warehouse = 1
    -- 'Chewy' for core network
    -- 'Chewy Pharmacy' for pharmacy
    -- 'Chewy Healthcare Services' for healthcare 
    AND product_company_description = 'Chewy'
    AND location_code NOT IN ('ITM0') 
UNION
   SELECT location_key,
          TRIM(location_code) -- fulfillment center's name
   FROM
    chewybi.locations
   WHERE TRIM(location_code) IN ('WFC2') -- hardcoded for including WFC2 in the early part of the year
;

--some order_ids repeat on a regular basis. There are about 1000 such ids. They have no shipment tracking number associated
-- order_id = 1292264392
-- select last batch of an order_id
DROP TABLE IF Exists order_batch_details;
CREATE LOCAL TEMP TABLE order_batch_details ON COMMIT PRESERVE ROWS AS
with batch_dt as (select distinct NEW_TIME(batch_dttm,'UTC','America/New_York') as batch_dttm,
                batch_id,
                order_id,
                zipcode,
                row_number() over (partition by order_id order by batch_dttm desc) as rn_ordbtch -- an order might get processed many times hence pick the last time it was processed.
                -- risk of an being paritially processed over multiple batches
from ors.order_items_to_route
where NEW_TIME(batch_dttm,'UTC','America/New_York') >= to_timestamp(:start_date_dttm, 'YYYY-MM-DD HH:MI:SS') 
      and NEW_TIME(batch_dttm,'UTC','America/New_York') < to_timestamp(:end_date_dttm, 'YYYY-MM-DD HH:MI:SS') 
      and itemtype IN ('NORMAL')
--and order_id = 1312721293
--where date(NEW_TIME(batch_dttm,'UTC','America/New_York')) between :start_date and :end_date and itemtype IN ('NORMAL')
)
select batch_dttm, 
       batch_id,
       order_id,
       rn_ordbtch,
       zipcode
from batch_dt
where rn_ordbtch = 1
;

DROP TABLE IF Exists filtered_shp_transactions;
CREATE LOCAL TEMPORARY TABLE filtered_shp_transactions ON COMMIT PRESERVE ROWS AS
SELECT *
FROM(
SELECT ROW_NUMBER() OVER (PARTITION BY s.shipment_tracking_number) as rn, 
       s.shipment_tracking_number,
       s.shipment_planned_weight,
       s.order_id,
       DATE(s.shipment_shipped_dttm) as actual_ship_date,
       s.shipment_quantity,
       s.actual_transit_days,
       order_placed_dttm,
       TRIM(s.ffmcenter_name) as ffmcenter_name,
       s.carrier_code,
       SUM(s.shipment_quantity) OVER (PARTITION BY s.shipment_tracking_number) as shipment_count_of_items_in_box,
       p.product_length*p.product_width*p.product_height*s.shipment_quantity as product_vol
FROM chewybi.shipment_transaction_details s
     INNER JOIN core_locations l
        ON l.location_code = s.ffmcenter_name
     INNER JOIN chewybi.products p 
        ON s.product_part_number = p.product_part_number
WHERE DATE(order_placed_dttm) BETWEEN TIMESTAMPADD (day, -10, (:start_date)) AND :end_date
AND cartonization_flag = 'true' AND s.product_company_description = 'Chewy'
) sq
WHERE sq.rn = 1
;

DROP TABLE IF Exists filtered_shp_transactions2;
CREATE LOCAL TEMPORARY TABLE filtered_shp_transactions2 ON COMMIT PRESERVE ROWS AS
select o.batch_dttm,
       o.batch_id,
       s.*
from filtered_shp_transactions s 
inner join order_batch_details o
on o.order_id = s.order_id
;

DROP TABLE IF Exists fedex_data;
CREATE LOCAL TEMPORARY TABLE fedex_data ON COMMIT PRESERVE ROWS AS
with fdx as (
  select *
  from chewybi.fedex_invoice_measures f
  where date(fedex_invoice_order_placed_dttm) between TIMESTAMPADD (day, -10, (:start_date)) AND TIMESTAMPADD (day, 10, (:end_date))
)
select *
from(
SELECT
    ROW_NUMBER() OVER (PARTITION BY s.shipment_tracking_number ORDER BY zeroifnull(f.fedex_invoice_freight_amt) + zeroifnull(f.fedex_invoice_fuel_surcharge_amount)+
                                                                        zeroifnull(f.fedex_invoice_residential_delivery_amount) + zeroifnull(f.fedex_invoice_additional_handling_amount) desc) as row_num_inv,  
    s.shipment_tracking_number,
    s.ffmcenter_name AS fc_name,
    'fedex' AS carrier,
    (CASE WHEN carrier_code IN ('FDXGD', 'FDXHD') THEN 1 ELSE 0 END) AS ground_mode,
    --f.fedex_invoice_net_charge_amt AS cost,
    f.fedex_invoice_freight_amt as base_cost,
    --zeroifnull(f.fedex_invoice_fuel_surcharge_amount) as fuel_surch,
    f.fedex_invoice_fuel_surcharge_amount as fuel_surch,
    --zeroifnull(f.fedex_invoice_residential_delivery_amount) as residential_surch,
    f.fedex_invoice_residential_delivery_amount as residential_surch,
    --zeroifnull(f.fedex_invoice_additional_handling_amount) as handling_surch,
    f.fedex_invoice_additional_handling_amount + 
    CASE WHEN fedex_invoice_miscellaneous_service_charge1_code_key = 2869 THEN fedex_invoice_miscellaneous1_amount ELSE 0 END as handling_surch,
    (f.fedex_invoice_freight_amt+zeroifnull(f.fedex_invoice_fuel_surcharge_amount)+
              zeroifnull(f.fedex_invoice_residential_delivery_amount)+zeroifnull(f.fedex_invoice_additional_handling_amount)+ 
              zeroifnull(f.fedex_invoice_earned_discount) + zeroifnull(f.fedex_invoice_performance_pricing_discount)+
              CASE WHEN fedex_invoice_miscellaneous_service_charge1_code_key = 2869 THEN fedex_invoice_miscellaneous1_amount ELSE 0 END)  as cost, --fedex code 2869 is for oversize surcharge 
    f.fedex_invoice_bill_weight AS wt,
    fedex_invoice_original_weight as invoice_actual_weight,
    CAST(f.fedex_invoice_length*f.fedex_invoice_width*f.fedex_invoice_height/166 AS INT) AS dim_weight
    --CAST(f.order_id, int) AS order_id_shpt,
    --f.fedex_invoice_fuel_surcharge_amount AS fuel_surch
FROM
    filtered_shp_transactions2 s
    INNER JOIN 
        fdx f
        ON f.shipment_tracking_number = s.shipment_tracking_number
    INNER JOIN
        core_locations l
        ON l.location_code = s.ffmcenter_name
WHERE
    DATE(s.batch_dttm) BETWEEN :start_date AND :end_date AND
    f.fedex_invoice_bill_to_account NOT IN (
        '912438155',
        '584064706',
        '617400146',
        '951701297',
        '618122425'
    )
    AND s.carrier_code IN ('FDXGD', 'FDXON', 'FDXHD', 'FDXST', 'FDX2D')
) sq
where sq.row_num_inv = 1 
;

DROP TABLE IF Exists ontrac_data;
CREATE LOCAL TEMPORARY TABLE ontrac_data ON COMMIT PRESERVE ROWS AS
with ontrc as (
  select *
  from chewybi.ontrac_invoice_measures
  where ontrac_invoice_date between TIMESTAMPADD (day, -10, (:start_date)) AND TIMESTAMPADD (day, 10, (:end_date))
)
select *
from (
SELECT 
    ROW_NUMBER() OVER (PARTITION BY s.shipment_tracking_number ORDER BY zeroifnull(o.ontrac_invoice_ground_service_charge_cost) + zeroifnull(o.ontrac_invoice_fuel_surcharge) +
                                                                        zeroifnull(o.ontrac_invoice_residential_fee) + 
                                                                        zeroifnull(o.ontrac_invoice_heavy_weight_service_charge_cost) + zeroifnull(o.ontrac_invoice_oversize_fee) desc) as row_num_inv,
    s.shipment_tracking_number,
    s.ffmcenter_name AS fc_name,
    'ontrac' AS carrier, 
    1 AS ground_mode,
    --o.ontrac_invoice_total_charge AS cost,
    o.ontrac_invoice_ground_service_charge_cost as base_cost,
    --zeroifnull(o.ontrac_invoice_fuel_surcharge) as fuel_surch,
    o.ontrac_invoice_fuel_surcharge as fuel_surch,
    --zeroifnull(o.ontrac_invoice_residential_fee) as residential_surch,
    o.ontrac_invoice_residential_fee as residential_surch,
    zeroifnull(o.ontrac_invoice_heavy_weight_service_charge_cost) + zeroifnull(o.ontrac_invoice_oversize_fee) as handling_surch,
    o.ontrac_invoice_ground_service_charge_cost + zeroifnull(o.ontrac_invoice_fuel_surcharge) +
               zeroifnull(o.ontrac_invoice_residential_fee) + 
               zeroifnull(o.ontrac_invoice_heavy_weight_service_charge_cost) + zeroifnull(o.ontrac_invoice_oversize_fee) as cost,
    o.ontrac_invoice_weight AS wt,
    0.0 as invoice_actual_weight,
    CAST(o.ontrac_invoice_package_length*o.ontrac_invoice_package_width*o.ontrac_invoice_package_height AS INT) AS dim_weight
    --CAST(o.ontrac_invoice_order_id, int) AS order_id_shpt,
    --o.ontrac_invoice_fuel_surcharge AS fuel_surch
FROM
    filtered_shp_transactions2 s
    INNER JOIN ontrc o
        ON o.ontrac_invoice_shipment_tracking_number = s.shipment_tracking_number 
    INNER JOIN
        core_locations l
        ON l.location_code = s.ffmcenter_name
WHERE
    DATE(s.batch_dttm) BETWEEN :start_date AND :end_date AND
    s.carrier_code = 'ONTRGD'
) sq
where sq.row_num_inv = 1
;

DROP TABLE IF Exists usps_data;
CREATE LOCAL TEMPORARY TABLE usps_data ON COMMIT PRESERVE ROWS AS
SELECT *
FROM(
SELECT
    ROW_NUMBER() OVER (PARTITION BY s.shipment_tracking_number ORDER BY amount desc) as row_num_inv,
    s.shipment_tracking_number,
    s.ffmcenter_name AS fc_name,
    'usps' AS carrier,
    1 AS ground_mode,
    u.amount AS base_cost,
    0.0 as fuel_surch,
    0.0 as residential_surch,
    0.0 as handling_surch,
    u.amount as cost,
    u.weight AS wt,
    0.0 as invoice_actual_weight,
    0.0 AS dim_weight
    --CAST(u.order_id, int) AS order_id_shpt,
    --0.0 AS fuel_surch
FROM
    filtered_shp_transactions2 s
    INNER JOIN 
        chewybi.usps_usage_measures u
        ON u.shipment_tracking_number = s.shipment_tracking_number
    INNER JOIN
        core_locations l
        ON l.location_code = s.ffmcenter_name
WHERE
    --AND s.actual_ship_date BETWEEN :start_date AND TIMESTAMPADD (day, 4, (:end_date))
    DATE(s.batch_dttm) BETWEEN :start_date AND :end_date
    AND s.carrier_code IN ('USPS1C', 'FIRST', 'USPS1CHJ')
) sq
where sq.row_num_inv = 1
;

DROP TABLE IF Exists invoice;
CREATE LOCAL TEMPORARY TABLE invoice ON COMMIT PRESERVE ROWS AS (
    SELECT
        *
    FROM 
        fedex_data
    UNION
    SELECT
        *
    FROM
        ontrac_data
    UNION
    SELECT
        *
    FROM
        usps_data
);

DROP TABLE IF Exists predicted;
CREATE LOCAL TEMPORARY TABLE predicted ON COMMIT PRESERVE ROWS AS 
SELECT 
    tracking_number AS shipment_tracking_number,
    TRIM(FC) AS fc_name,
    (
        CASE 
            WHEN carrier IN ('FDXGD', 'FDXON', 'FDXHD', 'FDXST', 'FDX2D') 
                THEN 'fedex'
            WHEN carrier IN ('ONTRGD')
                THEN 'ontrac'
            WHEN carrier IN ('USPS1C', 'FIRST', 'USPS1CHJ')
                THEN 'usps'
            ELSE
                'unknown'
        END
    ) AS carrier,
    base_rate AS predicted_base_cost,
    zeroifnull(fuel) as predicted_fuel_surch,
    zeroifnull(resi_amount) as predicted_residential_surch,
    zeroifnull(handling_weight_amt) + zeroifnull(oversize_amount) as predicted_handling_surch,  
    base_rate + zeroifnull(fuel) +
        zeroifnull(resi_amount) + zeroifnull(handling_weight_amt) + zeroifnull(oversize_amount) as cost,
    bill_weight_oversize_included AS predicted_wt,
    actual_weight, 
    dim_weight,
    ZEROIFNULL(zone) as zone,
    height*width*length as vol,
    Oversize_flag as oversize_flag,
    add_handling_flag as handling_flag
FROM
    --sandbox_supply_chain.obt_predictive_cpp_2022
    sandbox_supply_chain.obt_predictive_cpp p
WHERE
    segment NOT IN ('Dropship', 'Pharmacy')
    AND carrier IN (
        'FDXGD', 'FDXON', 'FDXHD', 'FDXST', 'FDX2D', 
        'ONTRGD',
        'USPS1C', 'FIRST', 'USPS1CHJ'
    )
    AND ship_date BETWEEN TIMESTAMPADD (day, -10, (:start_date)) AND TIMESTAMPADD (day, 10, (:end_date)) 
;

DROP TABLE IF Exists cpos_pre;
CREATE LOCAL TEMPORARY TABLE cpos_pre ON COMMIT PRESERVE ROWS AS
SELECT  DATE(s.batch_dttm) as batch_date,
        HOUR(s.batch_dttm) as batch_hour,
        s.batch_id as batch_id,
        DATE(s.order_placed_dttm) as order_date,
        s.order_id,
        MAX(s.actual_ship_date) OVER (PARTITION BY s.order_id) as max_ship_date,
        s.shipment_tracking_number,
        --COUNT(s.shipment_tracking_number) OVER (PARTITION BY s.order_id) as num_shipments_order,
        SUM(CASE WHEN i.cost is NULL THEN 0 ELSE 1 END) OVER (PARTITION BY s.order_id) as num_shipments_order,
        (CASE WHEN s.ffmcenter_name IN ('AVP2','BNA1','MCI1') THEN 1.0*((CASE WHEN i.cost IS NULL THEN 0 ELSE shipment_count_of_items_in_box END))
             WHEN s.ffmcenter_name IN ('RNO1','MDT1') THEN 1.05*(CASE WHEN i.cost IS NULL THEN 0 ELSE shipment_count_of_items_in_box END)
             ELSE 1.1*(CASE WHEN i.cost IS NULL THEN 0 ELSE shipment_count_of_items_in_box END) END) as VCPU,
        i.cost AS cost,--+ (CASE WHEN p.zone=1 THEN 0.79 ELSE 0 END) 
        p.cost as predicted_cost,--+ (CASE WHEN p.zone=1 THEN 0.79 ELSE 0 END)
        SUM(CASE WHEN p.cost is NULL THEN 0 ELSE 1 END) OVER (PARTITION BY s.order_id) as num_shipments_order_pred,
        (CASE WHEN p.fc_name IN ('AVP2','BNA1','MCI1') THEN 1.0*(shipment_count_of_items_in_box)
             WHEN p.fc_name IN ('RNO1','MDT1') THEN 1.05*(shipment_count_of_items_in_box)
             ELSE 1.1*(shipment_count_of_items_in_box) END) as VCPU_pred,
        (CASE WHEN i.base_cost is NULL THEN 1 ELSE 0 END) as use_pred, 
	MIN(s.order_placed_dttm) OVER (PARTITION BY s.order_id) as date_order,
	i.fc_name,
        (CASE WHEN i.cost is NULL THEN NULL ELSE i.carrier END) as carrier,
        (CASE WHEN p.cost is NULL THEN NULL ELSE p.carrier END) as carrier_pred,
        CASE WHEN i.cost is NULL THEN 0 ELSE i.wt END as invoice_wt, 
        p.predicted_wt as predicted_wt,
        (CASE WHEN i.cost is NULL THEN 0 WHEN i.carrier in ('usps') THEN p.dim_weight ELSE i.dim_weight END) as dim_weight_invoice,
        p.dim_weight as dim_weight_pred,
        --COALESCE(i.wt,p.predicted_wt) as wt,
        (CASE 
             --WHEN i.carrier = 'fedex' THEN COALESCE(invoice_actual_weight,shipment_planned_weight,actual_weight)
             WHEN i.cost is NULL THEN 0 
             WHEN i.carrier = 'fedex' THEN invoice_actual_weight 
             ELSE COALESCE(actual_weight, shipment_planned_weight) END
        ) as actual_weight,
        COALESCE(actual_weight,shipment_planned_weight) as actual_weight_pred,
        --(CASE 
        --     WHEN i.carrier = 'usps' THEN p.vol 
        --     ELSE COALESCE(i.vol,p.vol) END
        --) as vol,
        p.vol as vol, 
        CASE WHEN i.cost is NOT NULL THEN COALESCE(i.fuel_surch, 0) ELSE COALESCE(predicted_fuel_surch, 0) END as fuel_surch,
        CASE WHEN i.cost is NOT NULL THEN COALESCE(i.handling_surch, 0) ELSE COALESCE(predicted_handling_surch, 0) END as handling_surch,
        CASE WHEN i.cost is NOT NULL THEN COALESCE(i.residential_surch, 0) ELSE COALESCE(predicted_residential_surch, 0) END as residential_surch,
        CASE WHEN i.cost is NOT NULL THEN COALESCE(i.base_cost, 0) ELSE COALESCE(p.predicted_base_cost, 0) END as base_cost,
        --CASE WHEN COALESCE(i.fuel_surch, predicted_fuel_surch) IS NULL THEN 0 ELSE COALESCE(i.fuel_surch, predicted_fuel_surch) END as fuel_surch,
        --CASE WHEN COALESCE(i.handling_surch, predicted_handling_surch) IS NULL THEN 0 ELSE COALESCE(i.handling_surch, predicted_handling_surch) END as handling_surch,
        --CASE WHEN COALESCE(i.residential_surch, predicted_residential_surch) IS NULL THEN 0 ELSE COALESCE(i.residential_surch, predicted_residential_surch) END as residential_surch,
        --CASE WHEN COALESCE(i.base_cost, p.predicted_base_cost) IS NULL THEN 0 ELSE COALESCE(i.base_cost, p.predicted_base_cost) END as base_cost,
        --MAX(p.zone) OVER (PARTITION BY s.order_id) as max_ship_zone,
        --AVG(p.zone) OVER (PARTITION BY s.order_id) as avg_ship_zone,
        MAX(CASE WHEN i.cost is NOT NULL THEN p.zone ELSE 0 END) OVER (PARTITION BY order_id) as max_ship_zone,
        AVG(CASE WHEN i.cost is NOT NULL THEN p.zone ELSE 0 END) OVER (PARTITION BY order_id) as avg_ship_zone,
        MAX(p.zone) OVER (PARTITION BY order_id) as max_ship_zone_exp,
        AVG(p.zone) OVER (PARTITION BY order_id) as avg_ship_zone_exp,
        --SUM(shipment_count_of_items_in_box) OVER (PARTITION BY s.order_id) as num_items_order,
        (CASE WHEN i.cost IS NULL THEN 0 ELSE shipment_count_of_items_in_box END) as num_items_order,
        shipment_count_of_items_in_box as num_items_order_pred,
        s.actual_transit_days as tnt,
        oversize_flag,
        handling_flag
FROM
        filtered_shp_transactions2 s
        INNER JOIN 
            predicted p
            ON p.shipment_tracking_number = s.shipment_tracking_number
        LEFT JOIN
            invoice i
            ON p.shipment_tracking_number = i.shipment_tracking_number
;

--DROP TABLE IF Exists zip_order_id;
--CREATE LOCAL TEMPORARY TABLE zip_order_id ON COMMIT PRESERVE ROWS AS
--select order_id,zipcode
--from (select order_id, 
--             zipcode, 
--             row_number() over (partition by order_id) as rn_zip 
--       from ors.order_items_to_route
--       where date(NEW_TIME(batch_dttm,'UTC','America/New_York')) between :start_date and :end_date and itemtype IN ('NORMAL')--,'UOM')
--       )s
--where s.rn_zip = 1
--;

--@set maxrows 10000000;
--@export on;
--@export set filename="C:\Users\gsunder\git_repos\ors2-event-estimation\data\cpo_singles_in_optimizer_data.csv";
--select sq.*,
--       z.zipcode
--from (select c.batch_date as date,
--       c.batch_id as batch_id,
--       c.order_id as order_id,
--       MAX(c.batch_hour) as batch_hour, 
--       SUM(c.cost) as cpo,
--       SUM(c.predicted_cost) as cpo_predicted,
--       SUM(c.shipment_inspector_cost) as cpo_shipment_inspector,
--       SUM(c.est_shipment_inspector_cost) as cpo_est_shipment_inspector,
--       SUM(c.cost)/AVG(num_shipments_order) as cpp,
--       SUM(c.predicted_cost)/AVG(num_shipments_order_pred) as cpp_predicted,
--       --SUM(c.shipment_inspector_cost)/ as cpp_shipment_inspector,
--       --SUM(c.est_shipment_inspector_cost) as cpp_est_shipment_inspector,
--       SUM(fuel_surch) as fuel_surch,
--       SUM(handling_surch) as handling_surch,
--       SUM(residential_surch) as residential_surch,
--       SUM(base_cost) as base_cost, 
--       --COUNT(DISTINCT c.shipment_tracking_number) as num_shipments_order,
--       AVG(num_shipments_order) as num_shipments_order,
--       AVG(num_shipments_order_pred) as num_shipments_order_pred,
--       SUM(CASE WHEN carrier = 'fedex' THEN 1 ELSE 0 END) as num_shipments_order_fedex,
--       SUM(CASE WHEN carrier = 'ontrac' THEN 1 ELSE 0 END) as num_shipments_order_ontrac,
--       SUM(CASE WHEN carrier_pred = 'fedex' THEN 1 ELSE 0 END) as num_shipments_order_pred_fedex,
--       SUM(CASE WHEN carrier_pred = 'ontrac' THEN 1 ELSE 0 END) as num_shipments_order_pred_ontrac,
--       SUM(invoice_wt) as ship_weight,
--       SUM(predicted_wt) as ship_weight_predicted,
--       SUM(si_wt) as ship_weight_si,
--       SUM(actual_weight) as actual_weight,
--       --MAX(tnt) as max_tnt,
--       AVG(max_ship_zone) as max_ship_zone,
--       AVG(avg_ship_zone) as avg_ship_zone,
--       AVG(max_ship_zone_exp) as max_ship_zone_exp,
--       AVG(avg_ship_zone_exp) as avg_ship_zone_exp,
--       --AVG(num_items_order) as num_items_order,
--       SUM(num_items_order) as UPO,
--       SUM(num_items_order_pred) as UPO_pred,
--       SUM(num_items_order_si) as UPO_si, 
--       SUM(num_items_order_si_est) as UPO_si_est,
--       SUM(use_pred) as num_null_invoice, 
--       --SUM(product_vol) as order_volume,
--       SUM(vol) as ship_volume,
--       (CASE WHEN SUM(oversize_flag)>1 THEN 1 ELSE 0 END) as oversize_flag,
--       (CASE WHEN SUM(handling_flag)>1 THEN 1 ELSE 0 END) as handling_flag
--      from cpos_pre c 
--      GROUP BY c.batch_date, c.batch_id,c.order_id) sq
--left join zip_order_id z
--     on z.order_id = sq.order_id  
--;

--with shipment_inspector2 as (
--    select *
--    from (
--    select order_id,
--           customer_postcode AS zipcode,
--           row_number() over (partition by order_id) as rn_zip
--    from shipment_inspector ) si
--    where si.rn_zip = 1
--)
--with order_det as (
--select order_id,zipcode
--from (select order_id, 
--             zipcode, 
--             row_number() over (partition by order_id) as rn_zip 
--       from ors.order_items_to_route
--       where NEW_TIME(batch_dttm,'UTC','America/New_York') >= to_timestamp(:start_date_dttm, 'YYYY-MM-DD HH:MI:SS') and NEW_TIME(batch_dttm,'UTC','America/New_York') < to_timestamp(:end_date_dttm, 'YYYY-MM-DD HH:MI:SS') and itemtype IN ('NORMAL')
--       )s
--where s.rn_zip = 1
--)
select sq.*,
       zipcode
from (
select c.batch_date as date,
       c.batch_id as batch_id,
       c.order_id as order_id,
       MAX(c.batch_hour) as batch_hour, 
       SUM(c.cost) as cpo,
       SUM(c.VCPU) as VCPO,
       SUM(c.cost)+SUM(c.VCPU) as tcpo,
       SUM(c.predicted_cost) as cpo_predicted,
       SUM(c.VCPU_pred) as VCPO_pred,
       SUM(c.predicted_cost)+SUM(c.VCPU_pred) as tcpo_pred,
       SUM(c.cost)/AVG(num_shipments_order) as cpp,
       (SUM(c.cost)+SUM(c.VCPU))/AVG(num_shipments_order) as tcpp,
       SUM(c.predicted_cost)/AVG(num_shipments_order_pred) as cpp_predicted,
       (SUM(c.predicted_cost)+SUM(c.VCPU_pred))/AVG(num_shipments_order_pred) as tcpp_predicted,
       --SUM(c.shipment_inspector_cost)/ as cpp_shipment_inspector,
       --SUM(c.est_shipment_inspector_cost) as cpp_est_shipment_inspector,
       SUM(fuel_surch) as fuel_surch,
       SUM(handling_surch) as handling_surch,
       SUM(residential_surch) as residential_surch,
       SUM(base_cost) as base_cost, 
       --COUNT(DISTINCT c.shipment_tracking_number) as num_shipments_order,
       AVG(num_shipments_order) as num_shipments_order,
       AVG(num_shipments_order_pred) as num_shipments_order_pred,
       SUM(CASE WHEN carrier = 'fedex' THEN 1 ELSE 0 END) as num_shipments_order_fedex,
       SUM(CASE WHEN carrier = 'ontrac' THEN 1 ELSE 0 END) as num_shipments_order_ontrac,
       SUM(CASE WHEN carrier_pred = 'fedex' THEN 1 ELSE 0 END) as num_shipments_order_pred_fedex,
       SUM(CASE WHEN carrier_pred = 'ontrac' THEN 1 ELSE 0 END) as num_shipments_order_pred_ontrac,
       SUM(invoice_wt) as ship_weight,
       SUM(predicted_wt) as ship_weight_predicted,
       SUM(actual_weight) as actual_weight,
       SUM(actual_weight_pred) as actual_weight_pred,
       SUM(dim_weight_invoice) as dim_weight,
       SUM(dim_weight_pred) as dim_weight_pred,
       --MAX(tnt) as max_tnt,
       AVG(max_ship_zone) as max_ship_zone,
       AVG(avg_ship_zone) as avg_ship_zone,
       AVG(max_ship_zone_exp) as max_ship_zone_exp,
       AVG(avg_ship_zone_exp) as avg_ship_zone_exp,
       --AVG(num_items_order) as num_items_order,
       SUM(num_items_order) as UPO,
       SUM(num_items_order_pred) as UPO_pred,
       SUM(use_pred) as num_null_invoice, 
       --SUM(product_vol) as order_volume,
       SUM(vol) as ship_volume,
       (CASE WHEN SUM(oversize_flag)>1 THEN 1 ELSE 0 END) as oversize_flag,
       (CASE WHEN SUM(handling_flag)>1 THEN 1 ELSE 0 END) as handling_flag
from cpos_pre c 
GROUP BY c.batch_date, c.batch_id,c.order_id ) sq
--left join shipment_inspector2 si
left join order_batch_details si
on sq.order_id = si.order_id
;

--@export off
--@set maxrows 10000000;
--@export on;
--@export set filename="C:\Users\gsunder\git_repos\ors2-event-estimation\data\cpo_IAC_data.csv";
--select c.batch_date as date,
--       c.batch_id as batch_id,
--       c.order_id as order_id,
--       SUM(c.cost) as cpo,
--       SUM(fuel_surch) as fuel_surch,
--       SUM(handling_surch) as handling_surch,
--       SUM(residential_surch) as residential_surch,
--       SUM(base_cost) as base_cost, 
--       COUNT(DISTINCT c.shipment_tracking_number) as num_shipments_order,
--       AVG(num_shipments_order) as pre_computed_num_shipments_order,
--       SUM(wt) as wt,
--       SUM(actual_weight) as actual_weight,
--       MAX(tnt) as max_tnt,
--       AVG(max_ship_zone) as max_ship_zone,
--       AVG(ground_mode) as ground_mode,
--       AVG(num_items_order) as num_items_order 
--from cpos_pre c 
--GROUP BY c.batch_date, c.batch_id,c.order_id
--;

