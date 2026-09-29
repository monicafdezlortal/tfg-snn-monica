import mne

# Carga uno de los archivos .cnt para obtener los nombres de los 62 canales
raw = mne.io.read_raw_cnt(
    r"SEED\SEED_EEG\SEED_RAW_EEG\1_1.cnt",
    preload=False,
    verbose=False
)

channel_names = raw.ch_names
print(f"Total canales: {len(channel_names)}")
print()
print("Lista completa (índice: nombre):")
for i, name in enumerate(channel_names):
    print(f"  {i:2d}: {name}")

print()
target = ["FT7", "FT8", "T7", "T8"]
print("Índices de los canales de interés:")
for name in target:
    matches = [(i, n) for i, n in enumerate(channel_names) if n.upper() == name.upper()]
    if matches:
        print(f"  {name}: índice {matches[0][0]}")
    else:
        print(f"  {name}: NO ENCONTRADO — nombres similares: {[n for n in channel_names if name[:2].upper() in n.upper()]}")
