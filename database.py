import json
import time
import io

import numpy as np
from cryptography.fernet import Fernet

from config import DATABASE_PATH, SECRET_KEY_PATH


def get_or_create_key():
    if SECRET_KEY_PATH.exists():
        return SECRET_KEY_PATH.read_bytes()

    key = Fernet.generate_key()
    SECRET_KEY_PATH.write_bytes(key)
    return key


def encrypt_bytes(data: bytes) -> bytes:
    fernet = Fernet(get_or_create_key())
    return fernet.encrypt(data)


def decrypt_bytes(data: bytes) -> bytes:
    fernet = Fernet(get_or_create_key())
    return fernet.decrypt(data)


def l2_normalize(vector):
    vector = np.asarray(vector, dtype=np.float32)
    vector_norm = np.linalg.norm(vector)

    if vector_norm == 0:
        return vector

    return vector / vector_norm


def cosine_similarity(a, b):
    a = l2_normalize(a)
    b = l2_normalize(b)

    return float(np.dot(a, b))


def create_profile(name, embeddings):
    if not embeddings:
        return None

    normalized_embeddings = [
        l2_normalize(embedding)
        for embedding in embeddings
    ]

    embedding_matrix = np.vstack(normalized_embeddings)

    return {
        "name": name,
        "created_at": time.time(),
        "global_embedding": l2_normalize(
            np.mean(embedding_matrix, axis=0)
        ),
        "samples": embedding_matrix,
    }


def save_database(profiles):
    arrays = {}
    metadata = []

    for index, profile in enumerate(profiles):
        user_id = f"user_{index:03d}"

        metadata.append(
            {
                "user_id": user_id,
                "name": profile["name"],
                "created_at": profile["created_at"],
            }
        )

        arrays[f"{user_id}_global_embedding"] = profile["global_embedding"]
        arrays[f"{user_id}_samples"] = profile["samples"]

    arrays["metadata"] = np.array(
        json.dumps(metadata),
        dtype=np.str_,
    )

    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)

    encrypted_data = encrypt_bytes(buffer.getvalue())
    DATABASE_PATH.write_bytes(encrypted_data)


def load_database():
    if not DATABASE_PATH.exists():
        return []

    try:
        encrypted_data = DATABASE_PATH.read_bytes()
        decrypted_data = decrypt_bytes(encrypted_data)

        buffer = io.BytesIO(decrypted_data)
        data = np.load(buffer, allow_pickle=False)

    except Exception as error:
        print("Could not load encrypted database:", error)
        return []

    if "metadata" not in data.files:
        return []

    metadata = json.loads(str(data["metadata"]))
    profiles = []

    for item in metadata:
        user_id = item["user_id"]

        global_key = f"{user_id}_global_embedding"
        samples_key = f"{user_id}_samples"

        if global_key not in data.files or samples_key not in data.files:
            continue

        profiles.append(
            {
                "name": item["name"],
                "created_at": item["created_at"],
                "global_embedding": data[global_key],
                "samples": data[samples_key],
            }
        )

    return profiles


def match_database(profiles, embedding):
    if not profiles:
        return None, 0.0, 0.0

    embedding = l2_normalize(embedding)

    results = []

    for profile in profiles:
        global_score = cosine_similarity(
            profile["global_embedding"],
            embedding,
        )

        sample_scores = [
            cosine_similarity(sample, embedding)
            for sample in profile["samples"]
        ]

        sample_score = max(sample_scores) if sample_scores else 0.0
        score = max(global_score, sample_score)

        results.append((profile, score))

    results.sort(key=lambda item: item[1], reverse=True)

    best_profile, best_score = results[0]
    second_score = results[1][1] if len(results) > 1 else 0.0

    return best_profile, best_score, second_score