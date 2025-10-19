import json
import pickle
import argparse
import yaml
import logging
from tqdm import tqdm
from beaver import BeaverDB, Document

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

DEFAULT_CONFIG = {
    "collection": "images",
    "mode": "upsert",
}

def load_yaml_config(config_path: str) -> dict:
    if not config_path:
        return {}
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            logging.info(f"Loading configuration from: {config_path}")
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logging.warning(f"Configuration file not found at {config_path}. Ignoring.")
        return {}
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file '{config_path}': {e}")
        return {}

def load_index_data(file_path: str) -> list:
    data = []
    logging.info(f"Loading data from: {file_path}")
    try:
        if file_path.endswith(".jsonl"):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
        elif file_path.endswith((".pkl", ".pickle")):
            with open(file_path, 'rb') as f:
                while True:
                    try:
                        data.append(pickle.load(f))
                    except EOFError:
                        break
        else:
            logging.error(f"Unsupported file format: {file_path}")
            return []
    except FileNotFoundError:
        logging.error(f"File not found at: {file_path}")
        return []
    except Exception as e:
        logging.error(f"An error occurred while reading the file: {e}")
        return []
    logging.info(f"Successfully loaded {len(data)} records.")
    return data

def run_import(config: dict):
    index_file = config.get("index_file")
    db_file = config.get("db_file")
    collection_name = config.get("collection")
    mode = config.get("mode", DEFAULT_CONFIG["mode"])

    if not index_file or not db_file:
        logging.error("--index-file and --db-file must be provided (either via CLI or config file).")
        return

    index_data = load_index_data(index_file)
    if not index_data:
        return

    logging.info(f"Opening database at: {db_file}")
    db = BeaverDB(db_file)
    image_collection = db.collection(collection_name)
    
    if mode == "rebuild":
        logging.info(f"Rebuilding collection '{collection_name}'...")
        count = 0
        for doc in image_collection:
            image_collection.drop(doc)
            count += 1
        logging.info(f"Removed {count} existing documents.")
        image_collection.compact(block=True)
    
    db_ids = set()
    if mode in ["insert-only", "update-only", "sync", "upsert"]:
        logging.info(f"Loading existing document IDs from '{collection_name}'...")
        db_ids = set(d.id for d in image_collection)
        logging.info(f"Found {len(db_ids)} existing documents in the collection.")
        
    index_ids = set()
    
    logging.info(f"Starting import/update process in '{mode}' mode for {len(index_data)} documents...")
    
    inserted_count = 0
    updated_count = 0
    skipped_count = 0
    
    for item in tqdm(index_data, desc="Processing images"):
        vector = item.get("vector")
        metadata = item.get("metadata", {})
        doc_id = metadata.get("path")

        if not vector or not doc_id:
            logging.warning("Skipping item with no vector or no 'path' in metadata.")
            skipped_count += 1
            continue
        
        index_ids.add(doc_id)
        doc_exists = doc_id in db_ids
        
        should_index = False
        
        if mode == "upsert" or mode == "sync":
            should_index = True
            if doc_exists:
                updated_count += 1
            else:
                inserted_count += 1
        elif mode == "insert-only":
            if not doc_exists:
                should_index = True
                inserted_count += 1
            else:
                skipped_count += 1
        elif mode == "update-only":
            if doc_exists:
                should_index = True
                updated_count += 1
            else:
                skipped_count += 1
        elif mode == "rebuild":
            should_index = True

        if should_index:
            doc = Document(id=doc_id, embedding=vector, **metadata)
            image_collection.index(doc)

    if mode == "sync":
        ids_to_drop = db_ids - index_ids
        logging.info(f"Found {len(ids_to_drop)} documents in DB missing from index file. Removing...")
        
        removed_count = 0
        for doc_id in tqdm(ids_to_drop, desc="Removing obsolete images"):
            image_collection.drop(doc_id)
            removed_count += 1
        
        final_db_ids = set(d.id for d in image_collection)
        
        if removed_count > 0:
            logging.info("Compacting collection after removal...")
            image_collection.compact(block=True)
        
        if len(final_db_ids) == len(index_ids):
             logging.info("Synchronization successful! The collection now mirrors the index file.")
        else:
             logging.warning("Synchronization completed, but final document count does not match index size.")

    logging.info("Process completed successfully! ✅")
    current_doc_count = len(image_collection)
    
    if mode == "rebuild":
        logging.info(f"Total documents indexed in '{collection_name}' collection: {current_doc_count}")
    elif mode == "upsert":
        logging.info(f"Summary: {inserted_count} inserted, {updated_count} updated, {skipped_count} skipped.")
        logging.info(f"Total documents in '{collection_name}' collection: {current_doc_count}")
    elif mode == "insert-only":
        logging.info(f"Summary: {inserted_count} inserted, {skipped_count} skipped (document existed).")
        logging.info(f"Total documents in '{collection_name}' collection: {current_doc_count}")
    elif mode == "update-only":
        logging.info(f"Summary: {updated_count} updated, {skipped_count} skipped (document new).")
        logging.info(f"Total documents in '{collection_name}' collection: {current_doc_count}")
    elif mode == "sync":
        inserted = len(final_db_ids.intersection(index_ids) - db_ids.intersection(index_ids))
        updated = len(db_ids.intersection(index_ids))
        logging.info(f"Summary: {inserted} inserted, {updated} updated, {removed_count} removed.")
        logging.info(f"Total documents in '{collection_name}' collection: {current_doc_count}")

    db.close()


def main():
    parser = argparse.ArgumentParser(
        description="Imports an image index into a BeaverDB database.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("-c", "--config", help="Path to a YAML configuration file.")
    parser.add_argument("--index-file", help="Path to the index file (.jsonl or .pkl).")
    parser.add_argument("--db-file", help="Path to the BeaverDB database file.")
    parser.add_argument("--collection", help="Name of the collection within the database.")
    
    parser.add_argument(
        "--mode",
        choices=["upsert", "rebuild", "insert-only", "update-only", "sync"],
        default=DEFAULT_CONFIG["mode"],
        help="Import strategy: 'upsert' (default), 'rebuild' (clear and insert all), 'insert-only' (skip existing), 'update-only' (skip new), 'sync' (upsert and remove missing from DB)."
    )

    args = parser.parse_args()
    
    config = DEFAULT_CONFIG.copy()

    if args.config:
        yaml_config = load_yaml_config(args.config)
        config.update(yaml_config)

    cli_args_provided = {k: v for k, v in vars(args).items() if v is not None and v != parser.get_default(k)}
    
    config.update(cli_args_provided)
    
    run_import(config)

if __name__ == "__main__":
    main()