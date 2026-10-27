
import requests


def emotion_detector(text_to_analyze):
    """Analyze text and return its emotion scores and dominant emotion."""

    if not isinstance(text_to_analyze, str) or not text_to_analyze.strip():
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
        }

    url = (
        "https://sn-watson-emotion.labs.skills.network/"
        "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
    )

    headers = {
        "grpc-metadata-mm-model-id":
            "emotion_aggregated-workflow_lang_en_stock"
    }

    payload = {
        "raw_document": {
            "text": text_to_analyze
        }
    }

    try:
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=30,
        )

        if response.status_code == 400:
            return {
                "anger": None,
                "disgust": None,
                "fear": None,
                "joy": None,
                "sadness": None,
                "dominant_emotion": None,
            }

        response.raise_for_status()
        result = response.json()
        emotions = result["emotionPredictions"][0]["emotion"]

        scores = {
            emotion: emotions[emotion]
            for emotion in ("anger", "disgust", "fear", "joy", "sadness")
        }

        scores["dominant_emotion"] = max(
            scores, key=scores.get
        )

        return scores

    except (requests.RequestException, KeyError, IndexError, ValueError):
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
        }
