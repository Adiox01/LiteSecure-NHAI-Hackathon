import json
import time

from cryptography.fernet import Fernet

from config import SYNC_QUEUE_PATH, SECRET_KEY_PATH


def get_fernet():
    key = SECRET_KEY_PATH.read_bytes()
    return Fernet(key)


def load_queue():
    if not SYNC_QUEUE_PATH.exists():
        return []

    try:
        encrypted_data = SYNC_QUEUE_PATH.read_bytes()
        decrypted_data = get_fernet().decrypt(encrypted_data)
        return json.loads(decrypted_data.decode("utf-8"))
    except Exception as error:
        print("Could not load sync queue:", error)
        return []


def save_queue(queue):
    data = json.dumps(queue, indent=2).encode("utf-8")
    encrypted_data = get_fernet().encrypt(data)
    SYNC_QUEUE_PATH.write_bytes(encrypted_data)


def add_auth_event(name, score, status):
    queue = load_queue()

    queue.append(
        {
            "name": name,
            "score": round(float(score), 4),
            "status": status,
            "timestamp": time.time(),
            "synced": False,
        }
    )

    save_queue(queue)


def mock_sync_to_aws():
    queue = load_queue()

    if not queue:
        print("No pending sync events.")
        return

    print(f"Syncing {len(queue)} events to AWS mock server...")

    # Prototype simulation: assume upload success.
    for event in queue:
        print(
            f"Uploaded: {event['name']} "
            f"{event['status']} "
            f"{event['score']}"
        )

    # Purge local queue after successful sync.
    save_queue([])

    print("Sync complete. Local queue purged.")