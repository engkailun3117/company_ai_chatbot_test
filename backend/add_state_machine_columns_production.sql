-- Migration: Add state machine columns to company_onboarding (PRODUCTION)
-- This adds the columns needed for server-driven field collection.

-- WARNING: This will modify the PRODUCTION company_onboarding table!

-- Step 1: Create enum types (if they don't exist)

-- Drop existing enums if they exist (uncomment if needed)
-- DROP TYPE IF EXISTS onboardingstage CASCADE;
-- DROP TYPE IF EXISTS productfield CASCADE;

CREATE TYPE onboardingstage AS ENUM (
    'industry', 'capital_amount', 'invention_patent_count',
    'utility_patent_count', 'certification_count', 'esg_certification',
    'product', 'completed'
);

CREATE TYPE productfield AS ENUM (
    'product_id', 'product_name', 'price',
    'main_raw_materials', 'product_standard', 'technical_advantages'
);

-- Step 2: Add columns to company_onboarding table
ALTER TABLE company_onboarding
ADD COLUMN IF NOT EXISTS current_stage onboardingstage DEFAULT 'industry' NOT NULL;

ALTER TABLE company_onboarding
ADD COLUMN IF NOT EXISTS current_product_field productfield;

ALTER TABLE company_onboarding
ADD COLUMN IF NOT EXISTS current_product_draft TEXT;

-- Step 3: Update existing records to sync stage with data
UPDATE company_onboarding
SET current_stage = CASE
    WHEN industry IS NULL THEN 'industry'::onboardingstage
    WHEN capital_amount IS NULL THEN 'capital_amount'::onboardingstage
    WHEN invention_patent_count IS NULL THEN 'invention_patent_count'::onboardingstage
    WHEN utility_patent_count IS NULL THEN 'utility_patent_count'::onboardingstage
    WHEN certification_count IS NULL THEN 'certification_count'::onboardingstage
    WHEN esg_certification IS NULL THEN 'esg_certification'::onboardingstage
    ELSE 'product'::onboardingstage
END;

-- Done! Please restart the server after running this migration.
