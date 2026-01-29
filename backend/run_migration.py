"""
Run database migration to add state machine columns.
Execute this script once to add the new columns needed for the state machine.
"""

from sqlalchemy import text
from database import engine

def run_migration():
    """Add state machine columns to company_onboarding_test table"""

    migration_sql = """
    -- Step 1: Create the enum types (if they don't exist)
    DO $$ BEGIN
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
    EXCEPTION
        WHEN duplicate_object THEN null;
    END $$;

    DO $$ BEGIN
        CREATE TYPE productfield_test AS ENUM (
            'product_id',
            'product_name',
            'price',
            'main_raw_materials',
            'product_standard',
            'technical_advantages'
        );
    EXCEPTION
        WHEN duplicate_object THEN null;
    END $$;

    -- Step 2: Add the new columns
    ALTER TABLE company_onboarding_test
    ADD COLUMN IF NOT EXISTS current_stage onboardingstage_test DEFAULT 'industry' NOT NULL;

    ALTER TABLE company_onboarding_test
    ADD COLUMN IF NOT EXISTS current_product_field productfield_test;

    ALTER TABLE company_onboarding_test
    ADD COLUMN IF NOT EXISTS current_product_draft TEXT;

    -- Step 3: Update existing records to sync stage with data
    UPDATE company_onboarding_test
    SET current_stage = CASE
        WHEN industry IS NULL THEN 'industry'::onboardingstage_test
        WHEN capital_amount IS NULL THEN 'capital_amount'::onboardingstage_test
        WHEN invention_patent_count IS NULL THEN 'invention_patent_count'::onboardingstage_test
        WHEN utility_patent_count IS NULL THEN 'utility_patent_count'::onboardingstage_test
        WHEN certification_count IS NULL THEN 'certification_count'::onboardingstage_test
        WHEN esg_certification IS NULL THEN 'esg_certification'::onboardingstage_test
        ELSE 'product'::onboardingstage_test
    END
    WHERE current_stage = 'industry';
    """

    print("Running migration to add state machine columns...")

    with engine.connect() as conn:
        # Execute each statement separately for better error handling
        statements = [
            # Create enum types
            """
            DO $$ BEGIN
                CREATE TYPE onboardingstage_test AS ENUM (
                    'industry', 'capital_amount', 'invention_patent_count',
                    'utility_patent_count', 'certification_count', 'esg_certification',
                    'product', 'completed'
                );
            EXCEPTION WHEN duplicate_object THEN null;
            END $$;
            """,
            """
            DO $$ BEGIN
                CREATE TYPE productfield_test AS ENUM (
                    'product_id', 'product_name', 'price',
                    'main_raw_materials', 'product_standard', 'technical_advantages'
                );
            EXCEPTION WHEN duplicate_object THEN null;
            END $$;
            """,
            # Add columns
            "ALTER TABLE company_onboarding_test ADD COLUMN IF NOT EXISTS current_stage onboardingstage_test DEFAULT 'industry' NOT NULL;",
            "ALTER TABLE company_onboarding_test ADD COLUMN IF NOT EXISTS current_product_field productfield_test;",
            "ALTER TABLE company_onboarding_test ADD COLUMN IF NOT EXISTS current_product_draft TEXT;",
            # Update existing records
            """
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
            """
        ]

        for i, stmt in enumerate(statements, 1):
            try:
                conn.execute(text(stmt))
                print(f"  Step {i}/{len(statements)} completed")
            except Exception as e:
                print(f"  Step {i} error: {e}")

        conn.commit()

    print("\nMigration completed successfully!")
    print("You can now restart the server.")

if __name__ == "__main__":
    run_migration()
