"""
Run database migration to add state machine columns for PRODUCTION.
Execute this script once to add the new columns needed for the state machine.

This migrates the PRODUCTION table (company_onboarding), not the test table.
"""

from sqlalchemy import text
from database import engine

def run_migration():
    """Add state machine columns to company_onboarding table (PRODUCTION)"""

    print("Running PRODUCTION migration to add state machine columns...")
    print("=" * 60)
    print("WARNING: This will modify the PRODUCTION company_onboarding table!")
    print("=" * 60)

    # Step 1: Check and drop existing enum types if they have wrong values
    print("\nStep 1: Checking existing enum types...")
    with engine.connect() as conn:
        try:
            # Check if enum exists
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT 1 FROM pg_type WHERE typname = 'onboardingstage'
                );
            """))
            enum_exists = result.scalar()

            if enum_exists:
                print("  Enum 'onboardingstage' exists, dropping and recreating...")
                # Need to drop column first if it exists, then drop type
                conn.execute(text("""
                    ALTER TABLE company_onboarding
                    DROP COLUMN IF EXISTS current_stage;
                """))
                conn.execute(text("DROP TYPE IF EXISTS onboardingstage CASCADE;"))
            conn.commit()
            print("  Step 1a completed")
        except Exception as e:
            print(f"  Step 1a error (non-fatal): {e}")
            conn.rollback()

    with engine.connect() as conn:
        try:
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT 1 FROM pg_type WHERE typname = 'productfield'
                );
            """))
            enum_exists = result.scalar()

            if enum_exists:
                print("  Enum 'productfield' exists, dropping and recreating...")
                conn.execute(text("""
                    ALTER TABLE company_onboarding
                    DROP COLUMN IF EXISTS current_product_field;
                """))
                conn.execute(text("DROP TYPE IF EXISTS productfield CASCADE;"))
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
                CREATE TYPE onboardingstage AS ENUM (
                    'industry', 'capital_amount', 'invention_patent_count',
                    'utility_patent_count', 'certification_count', 'esg_certification',
                    'product', 'completed'
                );
            """))
            conn.commit()
            print("  Created 'onboardingstage' enum")
        except Exception as e:
            print(f"  onboardingstage: {e}")
            conn.rollback()

    with engine.connect() as conn:
        try:
            conn.execute(text("""
                CREATE TYPE productfield AS ENUM (
                    'product_id', 'product_name', 'price',
                    'main_raw_materials', 'product_standard', 'technical_advantages'
                );
            """))
            conn.commit()
            print("  Created 'productfield' enum")
        except Exception as e:
            print(f"  productfield: {e}")
            conn.rollback()

    # Step 3: Add columns
    print("\nStep 3: Adding columns...")
    with engine.connect() as conn:
        try:
            conn.execute(text("""
                ALTER TABLE company_onboarding
                ADD COLUMN IF NOT EXISTS current_stage onboardingstage
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
                ALTER TABLE company_onboarding
                ADD COLUMN IF NOT EXISTS current_product_field productfield;
            """))
            conn.commit()
            print("  Added 'current_product_field' column")
        except Exception as e:
            print(f"  current_product_field error: {e}")
            conn.rollback()

    with engine.connect() as conn:
        try:
            conn.execute(text("""
                ALTER TABLE company_onboarding
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
            """))
            conn.commit()
            print("  Updated existing records")
        except Exception as e:
            print(f"  Update error: {e}")
            conn.rollback()

    print("\n" + "=" * 60)
    print("PRODUCTION Migration completed!")
    print("Please restart the server.")

if __name__ == "__main__":
    confirm = input("This will modify the PRODUCTION database. Type 'yes' to continue: ")
    if confirm.lower() == 'yes':
        run_migration()
    else:
        print("Migration cancelled.")
