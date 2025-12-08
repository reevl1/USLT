import cv2
import mediapipe as mp
import numpy as np
import argparse
import csv
import os
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from PIL import ImageFont, ImageDraw, Image
import numpy as np


DATASET_CSV = "sign_letters_dataset.csv"
MODEL_PATH = "sign_letters_knn.pkl"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def extract_hand_features(hand_landmarks):

    points = []
    for lm in hand_landmarks.landmark:
        points.append([lm.x, lm.y, lm.z])
    points = np.array(points)

    # запястье
    wrist = points[0]
    rel = points - wrist

    # "размер" руки: расстояние от запястья до средней фаланги среднего пальца
    hand_size = np.linalg.norm(points[9] - wrist)
    if hand_size < 1e-6:
        hand_size = 1.0

    rel /= hand_size
    # превращаем в одномерный вектор
    return rel.flatten()


def collect_mode(letter):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Не удалось открыть камеру")
        return

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    print("Режим сбора данных.")
    print(f"Показывай жест для буквы '{letter}', нажимай 'c' для сохранения кадра.")
    print("Нажми 'q' для выхода.")

    # Проверим, есть ли файл
    file_exists = os.path.isfile(DATASET_CSV)

    with open(DATASET_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Если файл новый — запишем заголовки
        if not file_exists:
            header = [f"f{i}" for i in range(63)] + ["label"]
            writer.writerow(header)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Камера не отдает кадр")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

            cv2.putText(
                frame,
                f"Letter: {letter} | 'c' - save, 'q' - quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Collect data", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("c"):
                if result.multi_hand_landmarks:
                    hand_landmarks = result.multi_hand_landmarks[0]
                    features = extract_hand_features(hand_landmarks)
                    row = list(features) + [letter]
                    writer.writerow(row)
                    print("Сохранён один пример для буквы", letter)
                else:
                    print("Рука не найдена, пример не сохранён")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


def train_mode():
    if not os.path.isfile(DATASET_CSV):
        print(f"Нет файла {DATASET_CSV}. Сначала собери данные (--mode collect).")
        return

    X = []
    y = []

    with open(DATASET_CSV, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            *features, label = row
            X.append([float(v) for v in features])
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    if len(np.unique(y)) < 2:
        print("Нужно хотя бы 2 разных буквы для обучения.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy на тесте:", acc)
    print(classification_report(y_test, y_pred))

    joblib.dump(clf, MODEL_PATH)
    print(f"Модель сохранена в {MODEL_PATH}")


def draw_text_pil(frame, text, x=10, y=40, font_size=48, color=(0, 255, 0)):
    # frame: OpenCV BGR
    # text: str (любой Unicode, включая кириллицу)
    # color: (R, G, B)

    # convert OpenCV (BGR) → PIL (RGB)
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)

    draw.text((x, y), text, font=font, fill=(color[0], color[1], color[2]))

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)



def run_mode():
    if not os.path.isfile(MODEL_PATH):
        print(f"Нет модели {MODEL_PATH}. Сначала обучи её (--mode train).")
        return
    
    print("Загружаю модель...", flush=True)
    clf = joblib.load(MODEL_PATH)
    print("Модель загружена:", clf, flush=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Не удалось открыть камеру")
        return

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    print("Режим распознавания. 'q' — выход.")

    last_letter = None
    last_change_time = 0
    stable_letter = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Камера не отдает кадр")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        predicted_letter = " "

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            try:
                features = extract_hand_features(hand_landmarks).reshape(1, -1)
                predicted_letter = clf.predict(features)[0]
                print("Предсказал:", predicted_letter)
            except Exception as e:
                print("Ошибка предсказания:", e)
                predicted_letter = " "

        current_time = time.time()

        if predicted_letter == last_letter:
            if current_time - last_change_time >= 2.0:
                if len(stable_letter) == 0 or stable_letter[-1] != predicted_letter:
                    stable_letter += predicted_letter
                    print(">>> ЗАПОМНИЛ:", predicted_letter)
        else:
            last_letter = predicted_letter
            last_change_time = current_time

        frame = draw_text_pil(frame, f"Текущая: {predicted_letter}", x=10, y=10)
        frame = draw_text_pil(frame, f"Текст: {stable_letter}", x=10, y=70)

        cv2.imshow("Surdo", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()



def main():
    parser = argparse.ArgumentParser(
        description="Перевод букв с жестового языка на русский с использованием MediaPipe Hands"
    )
    parser.add_argument(
        "--mode",
        choices=["collect", "train", "run"],
        required=True,
        help="Режим работы: collect (сбор данных), train (обучение), run (распознавание)",
    )
    parser.add_argument(
        "--letter",
        type=str,
        help="Буква, которую собираем в режиме collect (например, А, Б, В...)",
    )

    args = parser.parse_args()

    if args.mode == "collect":
        if not args.letter:
            print("В режиме collect нужно указать букву: --letter А")
            return
        collect_mode(args.letter)
    elif args.mode == "train":
        train_mode()
    elif args.mode == "run":
        run_mode()


if __name__ == "__main__":
    main()
