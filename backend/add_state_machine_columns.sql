-- Migration: Add state machine columns to company_onboarding_test table
-- Run this SQL manually if the Python script doesn't work

-- Step 1: Drop existing columns and enum types (if they exist with wrong values)
ALTER TABLE company_onboarding_test DROP COLUMN IF EXISTS current_stage;
ALTER TABLE company_onboarding_test DROP COLUMN IF EXISTS current_product_field;
ALTER TABLE company_onboarding_test DROP COLUMN IF EXISTS current_product_draft;
DROP TYPE IF EXISTS onboardingstage_test CASCADE;
DROP TYPE IF EXISTS productfield_test CASCADE;

-- Step 2: Create the enum types
CREATE TYPE onboardingstage_test AS ENUM (
    'industry',
    'capital_amount',
    'invention_patent_count',
    'utility_patent_count',
    'certification_count',
    'esg_certification',
    'product',
    'completed'
);

CREATE TYPE productfield_test AS ENUM (
    'product_id',
    'product_name',
    'price',
    'main_raw_materials',
    'product_standard',
    'technical_advantages'
);

-- Step 3: Add the new columns
ALTER TABLE company_onboarding_test
ADD COLUMN current_stage onboardingstage_test DEFAULT 'industry' NOT NULL;

ALTER TABLE company_onboarding_test
ADD COLUMN current_product_field productfield_test;

ALTER TABLE company_onboarding_test
ADD COLUMN current_product_draft TEXT;

-- Step 4: Update existing records to sync stage with data
UPDATE company_onboarding_test
SET current_stage = CASE
    WHEN industry IS NULL THEN 'industry'::onboardingstage_test
    WHEN capital_amount IS NULL THEN 'capital_amount'::onboardingstage_test
    WHEN invention_patent_count IS NULL THEN 'invention_patent_count'::onboardingstage_test
    WHEN utility_patent_count IS NULL THEN 'utility_patent_count'::onboardingstage_test
    WHEN certification_count IS NULL THEN 'certification_count'::onboardingstage_test
    WHEN esg_certification IS NULL THEN 'esg_certification'::onboardingstage_test
    ELSE 'product'::onboardingstage_test
END;
