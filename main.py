import librosa
import numpy as np
from music21 import stream, note, meter, tempo
from pathlib import Path

# 경로 설정
AUDIO_PATH = "audio/test.mp3"
OUTPUT_FOLDER = "output/"

# Step 1: 음원 로드
def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

# Step 2: 박자 및 음정 추출
def analyze_audio(y, sr):
    # 박자 추출
    tempo_value, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)

    # 음정 추출
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_indices = np.argmax(magnitudes, axis=0)
    pitches = pitches[pitch_indices, range(magnitudes.shape[1])]
    pitches = [p if p > 0 else None for p in pitches]  # 유효한 음정만 남기기
    return tempo_value, beat_times, pitches

# Step 3: 음정을 MIDI 값으로 변환
def pitch_to_midi(pitches):
    midi_notes = []
    for pitch in pitches:
        if pitch:
            midi_note = librosa.hz_to_midi(pitch)
            midi_notes.append(int(round(midi_note)))
        else:
            midi_notes.append(None)
    return midi_notes

# Step 4: Music21으로 악보 생성
def create_sheet_music(tempo_value, beat_times, midi_notes):
    score = stream.Score()

    # 템포 설정
    part = stream.Part()
    part.append(tempo.MetronomeMark(number=int(tempo_value)))

    # 박자 설정
    part.append(meter.TimeSignature("4/4"))

    # 음표 추가
    for i, midi_note in enumerate(midi_notes):
        if midi_note is not None:
            new_note = note.Note(midi_note)
            new_note.quarterLength = 1  # 음표 길이 (4분음표)
            part.append(new_note)
        else:
            rest_note = note.Rest()
            rest_note.quarterLength = 1
            part.append(rest_note)

    score.append(part)

    # 결과 저장
    Path(OUTPUT_FOLDER).mkdir(exist_ok=True)
    score.write("musicxml", fp=f"{OUTPUT_FOLDER}/generated_sheet_music.xml")

# Main 실행
if __name__ == "__main__":
    y, sr = load_audio(AUDIO_PATH)
    tempo_value, beat_times, pitches = analyze_audio(y, sr)
    midi_notes = pitch_to_midi(pitches)
    create_sheet_music(tempo_value, beat_times, midi_notes)
