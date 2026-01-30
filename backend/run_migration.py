"""
Run database migration to add state machine columns.
Execute this script once to add the new columns needed for the state machine.
"""

from sqlalchemy import text
from database import engine

def run_migration():
    """Add state machine columns to company_onboarding_test table"""

    print("Running migration to add state machine columns...")
    print("=" * 50)

    # Step 1: Check and drop existing enum types if they have wrong values
    print("\nStep 1: Checking existing enum types...")
    with engine.connect() as conn:
        try:
            # Check if enum exists
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT 1 FROM pg_type WHERE typname = 'onboardingstage_test'
                );
            """))
            enum_exists = result.scalar()

            if enum_exists:
                print("  Enum 'onboardingstage_test' exists, dropping and recreating...")
                # Need to drop column first if it exists, then drop type
                conn.execute(text("""
                    ALTER TABLE company_onboarding_test
                    DROP COLUMN IF EXISTS current_stage;
                """))
                conn.execute(text("DROP TYPE IF EXISTS onboardingstage_test CASCADE;"))
            conn.commit()
            print("  Step 1a completed")
        except Exception as e:
            print(f"  Step 1a error (non-fatal): {e}")
            conn.rollback()

    with engine.connect() as conn:
        try:
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT 1 FROM pg_type WHERE typname = 'productfield_test'
                );
            """))
            enum_exists = result.scalar()

            if enum_exists:
                print("  Enum 'productfield_test' exists, dropping and recreating...")
                conn.execute(text("""
                    ALTER TABLE company_onboarding_test
                    DROP COLUMN IF EXISTS current_product_field;
                """))
                conn.execute(text("DROP TYPE IF EXISTS productfield_test CASCADE;"))
            conn.commit()
            print("  Step 1b completed")
        except Exception as e:
            print(f"  Step 1b error (non-fatal): {e}")
            conn.rollback()

    # Step 2: Create enum types
    print("\nStep 2: Creating enum types...")
    with engine.connect() as conn:
        try:
            conn.execute(text("""
                CREATE TYPE onboardingstage_test AS ENUM (
                    'industry', 'capital_amount', 'invention_patent_count',
                    'utility_patent_count', 'certification_count', 'esg_certification',
                    'product', 'completed'
                );
            """))
            conn.commit()
            print("  Created 'onboardingstage_test' enum")
        except Exception as e:
            print(f"  onboardingstage_test: {e}")
            conn.rollback()

    with engine.connect() as conn:
        try:
            conn.execute(text("""
                CREATE TYPE productfield_test AS ENUM (
                    'product_id', 'product_name', 'price',
                    'main_raw_materials', 'product_standard', 'technical_advantages'
                );
            """))
            conn.commit()
            print("  Created 'productfield_test' enum")
        except Exception as e:
            print(f"  productfield_test: {e}")
            conn.rollback()

    # Step 3: Add columns
    print("\nStep 3: Adding columns...")
    with engine.connect() as conn:
        try:
            conn.execute(text("""
                ALTER TABLE company_onboarding_test
                ADD COLUMN IF NOT EXISTS current_stage onboardingstage_test
                DEFAULT 'industry' NOT NULL;
            """))
            conn.commit()
            print("  Added 'current_stage' column")
        except Exception as e:
            print(f"  current_stage error: {e}")
            conn.rollback()

    with engine.connect() as conn:
        try:
            conn.execute(text("""
                ALTER TABLE company_onboarding_test
                ADD COLUMN IF NOT EXISTS current_product_field productfield_test;
            """))
            conn.commit()
            print("  Added 'current_product_field' column")
        except Exception as e:
            print(f"  current_product_field error: {e}")
            conn.rollback()

    with engine.connect() as conn:
        try:
            conn.execute(text("""
                ALTER TABLE company_onboarding_test
                ADD COLUMN IF NOT EXISTS current_product_draft TEXT;
            """))
            conn.commit()
            print("  Added 'current_product_draft' column")
        except Exception as e:
            print(f"  current_product_draft error: {e}")
            conn.rollback()

    # Step 4: Update existing records to sync stage with data
    print("\nStep 4: Syncing existing records...")
    with engine.connect() as conn:
        try:
            conn.execute(text("""
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
            """))
            conn.commit()
            print("  Updated existing records")
        except Exception as e:
            print(f"  Update error: {e}")
            conn.rollback()

    print("\n" + "=" * 50)
    print("Migration completed!")
    print("Please restart the server.")

if __name__ == "__main__":
    run_migration()
