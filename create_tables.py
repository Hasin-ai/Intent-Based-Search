import os
import logging
from sqlalchemy import create_engine, inspect
from dotenv import load_dotenv
from models import Product, Base

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_tables_if_not_exist():
    """Check if tables exist in database and create them if they don't"""
    try:
        # Load environment variables
        load_dotenv()
        
        # Get database URL
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL is not set in environment variables")
        
        # Create engine and inspector
        logger.info(f"Connecting to database...")
        engine = create_engine(database_url)
        inspector = inspect(engine)
        
        # Check if our table exists
        table_name = Product.__tablename__
        if inspector.has_table(table_name):
            logger.info(f"Table '{table_name}' already exists.")
        else:
            logger.info(f"Table '{table_name}' doesn't exist. Creating it now...")
            # Create all tables defined in Base.metadata
            Base.metadata.create_all(bind=engine)
            logger.info(f"Table '{table_name}' created successfully.")
        
        return True
    
    except Exception as e:
        logger.error(f"Error creating tables: {str(e)}")
        return False

if __name__ == "__main__":
    create_tables_if_not_exist()