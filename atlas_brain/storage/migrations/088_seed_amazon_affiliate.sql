-- Seed Amazon Associates affiliate partner row
INSERT INTO affiliate_partners (
    name, product_name, category, affiliate_url,
    commission_type, commission_value, enabled
) VALUES (
    'Amazon Associates',
    'Amazon',
    'consumer',
    'https://www.amazon.com?tag=atlas0e9b-20',
    'percentage',
    '1-10%',
    true
) ON CONFLICT ((lower(product_name))) DO NOTHING;
