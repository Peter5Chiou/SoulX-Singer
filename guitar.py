import pretty_midi

def add_strummed_chord(instrument, chord_notes, start_time, direction="down", strum_duration_ms=45, base_velocity=95):
    """
    chord_notes: 包含 (pitch, string_num) 的清單，例如 [(60, 5), (64, 4), (67, 3), (72, 2)]
    direction: "down" (下刷: 低音弦 -> 高音弦) 或 "up" (上刷: 高音弦 -> 低音弦)
    strum_duration_ms: 整個刷弦動作持續的總毫秒數
    """
    num_notes = len(chord_notes)
    if num_notes == 0:
        return

    # 依方向排序音符
    # 下刷：從低音弦(大弦號)刷到高音弦(小弦號)；上刷：從高音弦到低音弦
    reverse_sort = True if direction == "down" else False
    sorted_notes = sorted(chord_notes, key=lambda x: x[1], reverse=reverse_sort)

    # 計算每條弦之間的微小時間差
    delay_per_string_sec = (strum_duration_ms / 1000.0) / max(num_notes - 1, 1)

    # 上刷力道通常稍弱，下刷從大力漸弱
    current_velocity = base_velocity if direction == "down" else int(base_velocity * 0.85)
    velocity_decay = 3 if direction == "down" else 2

    for i, (pitch, string_num) in enumerate(sorted_notes):
        # 微調時間：第一音點在 start_time，後續音符依序微幅遞延
        note_start = start_time + (i * delay_per_string_sec)
        note_end = start_time + 1.5  # 音符持續時間 (可自訂)
        
        # 微調力道：加入 +/- 3 的隨機人聲感波動 (Humanization)
        import random
        final_velocity = max(40, min(127, current_velocity + random.randint(-3, 3)))

        note = pretty_midi.Note(
            velocity=final_velocity,
            pitch=pitch,
            start=note_start,
            end=note_end
        )
        instrument.notes.append(note)

        # 撥片越往後刷阻力越大，音量微幅遞減
        current_velocity -= velocity_decay

# 範例使用：
pm = pretty_midi.PrettyMIDI()
guitar = pretty_midi.Instrument(program=25) # 25 = Acoustic Guitar (steel)

# C 和弦 (t p3 p2 p1) -> (C3, 5弦), (G3, 3弦), (C4, 2弦), (E4, 1弦)
c_chord = [(48, 5), (55, 3), (60, 2), (64, 1)]

# 1. 0.0 秒時下刷 (Downstroke)
add_strummed_chord(guitar, c_chord, start_time=0.0, direction="down", strum_duration_ms=50)

# 2. 1.0 秒時上刷 (Upstroke)
add_strummed_chord(guitar, c_chord, start_time=1.0, direction="up", strum_duration_ms=35)

pm.instruments.append(guitar)
pm.write("strum_demo.mid")