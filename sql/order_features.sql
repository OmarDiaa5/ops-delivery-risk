-- One row per delivered order, using only what is known at purchase time.
-- Label: delivered after the date promised to the customer.
WITH items AS (
    SELECT oi.order_id,
           COUNT(*)                         AS n_items,
           COUNT(DISTINCT oi.seller_id)     AS n_sellers,
           SUM(oi.price)                    AS total_price,
           SUM(oi.freight_value)            AS total_freight,
           SUM(COALESCE(p.product_weight_g, 0)) AS total_weight_g,
           SUM(COALESCE(p.product_length_cm * p.product_height_cm * p.product_width_cm, 0)) AS total_volume_cm3,
           MIN(oi.seller_id)                AS main_seller_id,
           MIN(COALESCE(t.product_category_name_english, 'unknown')) AS main_category
    FROM order_items oi
    LEFT JOIN products p ON p.product_id = oi.product_id
    LEFT JOIN category_translation t ON t.product_category_name = p.product_category_name
    GROUP BY oi.order_id
),
pay AS (
    -- the payment type that covered most of the order value
    SELECT order_id, payment_type, installments FROM (
        SELECT order_id, payment_type,
               MAX(payment_installments) OVER (PARTITION BY order_id) AS installments,
               ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY payment_value DESC) AS rn
        FROM order_payments)
    WHERE rn = 1
)
SELECT o.order_id,
       o.order_purchase_timestamp                    AS purchase_ts,
       o.order_delivered_customer_date               AS delivered_ts,
       julianday(o.order_estimated_delivery_date) - julianday(o.order_purchase_timestamp) AS promised_days,
       c.customer_state,
       c.customer_zip_code_prefix                    AS customer_zip,
       s.seller_state,
       s.seller_zip_code_prefix                      AS seller_zip,
       CASE WHEN c.customer_state = s.seller_state THEN 1 ELSE 0 END AS same_state,
       i.main_seller_id                              AS seller_id,
       i.n_items, i.n_sellers, i.total_price, i.total_freight,
       i.total_freight / NULLIF(i.total_price, 0)    AS freight_ratio,
       i.total_weight_g, i.total_volume_cm3, i.main_category,
       COALESCE(pay.payment_type, 'unknown')         AS payment_type,
       COALESCE(pay.installments, 1)                 AS installments,
       CAST(strftime('%w', o.order_purchase_timestamp) AS INTEGER) AS purchase_dow,
       CAST(strftime('%H', o.order_purchase_timestamp) AS INTEGER) AS purchase_hour,
       CAST(strftime('%m', o.order_purchase_timestamp) AS INTEGER) AS purchase_month,
       CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1 ELSE 0 END AS is_late
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
JOIN items i     ON i.order_id = o.order_id
JOIN sellers s   ON s.seller_id = i.main_seller_id
LEFT JOIN pay    ON pay.order_id = o.order_id
WHERE o.order_status = 'delivered'
  AND o.order_delivered_customer_date IS NOT NULL
  AND o.order_estimated_delivery_date IS NOT NULL;
