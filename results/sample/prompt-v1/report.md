# Evaluationsbericht: Modellvergleich

## Konfiguration
- Fragen: `41`
- Retrieval: `hybrid_rerank`
- Top-k: `4`
- Embedding: `jinaai/jina-embeddings-v2-base-de`
- Reranker: `jinaai/jina-reranker-v2-base-multilingual`
- Chunking: `section`
- Judge: `–`

## Ergebnisse
| Modell | Genauigkeit | Fakten-Recall | Zitat korrekt | Korrekt abgelehnt | Fälschlich abgelehnt | Faithfulness (1–5) | Latenz p50 / p95 | Kosten / 1000 Fragen |
|---|---|---|---|---|---|---|---|---|
| qwen2.5-3b | **73 %** | 75 % | 94 % | 80 % | 0 % | – | 32.1 s / 49.4 s | $0.00 |
| llama3.2-3b | **59 %** | 64 % | 39 % | 60 % | 0 % | – | 26.6 s / 42.5 s | $0.00 |
| baseline-extractive | **56 %** | 69 % | 97 % | 0 % | 0 % | – | 0.0 s / 0.0 s | $0.00 |

**Genauigkeit**: beantwortbare Fragen enthalten alle erwarteten Fakten, unbeantwortbare Fragen werden abgelehnt.
**Faithfulness**: LLM-as-a-Judge, nur wenn mit `--judge` aktiviert.

## Genauigkeit nach Fragekategorie
| Modell | dokumentübergreifend | fakt | grenzfall | unbeantwortbar |
|---|---|---|---|---|
| baseline-extractive | 0 % | 76 % | 54 % | 0 % |
| qwen2.5-3b | 50 % | 86 % | 54 % | 80 % |
| llama3.2-3b | 0 % | 67 % | 54 % | 60 % |

## Fehlerbeispiele
- **baseline-extractive** · `hh-02` (fakt) – Bis zu welchem Betrag ist Bargeld in einem Wertschutzschrank versichert?
  > Wertsachen, die sich nicht in einem versperrten Möbel oder Wertschutzschrank befinden, sind insgesamt nur bis EUR 500 versichert. [1] Wertsache | in versperrtem Möbel | in einem Wertschutzschrank [1]
- **baseline-extractive** · `hh-03` (grenzfall) – Mein Fahrrad wurde um 23 Uhr vor dem Haus gestohlen, obwohl es an einem Laternenmast angeschlossen war. Ist das versichert?
  > Im Freien abgestellte Fahrräder sind nur versichert, wenn sie mit einem Schloss an einem fest verankerten Gegenstand angeschlossen waren und der Diebstahl zwischen 6:00 und 22:00 Uhr erfolgte. [1] Fahrräder sind gegen Diebstahl aus einem versperrten Raum (Wohnung, Kellerabteil, versperrter Fahrradra
- **baseline-extractive** · `hh-05` (grenzfall) – Sind Schäden durch Hochwasser in der Haushaltsversicherung gedeckt?
  > Überschwemmung, Hochwasser, Vermurung und Erdbeben; [1] Nicht versichert sind, soweit nicht ausdrücklich anders vereinbart, Schäden durch: [1]
- **baseline-extractive** · `hh-07` (fakt) – Wie hoch ist die Versicherungssumme der Privathaftpflicht?
  > Für Wertgegenstände wie Schmuck, Uhren, Kameras, Laptops und Mobiltelefone wird insgesamt höchstens 50 % der Versicherungssumme ersetzt. [3] Die Versicherungssumme beträgt EUR 2.000 je Person. [3]
- **baseline-extractive** · `hh-08` (grenzfall) – Mein Hund hat das Fahrrad meines Nachbarn beschädigt. Zahlt meine Haushaltsversicherung?
  > Versichert sind Schadenersatzverpflichtungen des Versicherungsnehmers und der im gemeinsamen Haushalt lebenden Personen aus den Gefahren des täglichen Lebens, beispielsweise als Fußgänger, Radfahrer, Sportler (ausgenommen Jagd) oder als Halter von Haustieren einschließlich Hunden. [1] Im Freien abge
- **baseline-extractive** · `hh-09` (fakt) – Ab welcher Windgeschwindigkeit gilt ein Sturm als versichertes Ereignis?
  > Kann eine gebuchte Reise nicht angetreten werden, ersetzt die Beispiel Versicherung AG die vertraglich geschuldeten Stornokosten, wenn die versicherte Person oder ein naher Angehöriger von einem der folgenden Ereignisse betroffen ist: [3] Versichert sind Schäden an den versicherten Sachen durch: [1]
- **baseline-extractive** · `rs-06` (grenzfall) – Kann ich eine Reise über den Stornoschutz stornieren, weil ich Angst vor Terroranschlägen am Reiseziel habe?
  > Stornierungen aus Angst vor Terroranschlägen, Unruhen oder Krankheiten am Reiseziel; [1] Die versicherte Person hat die Reise unverzüglich nach Eintritt des Stornogrundes zu stornieren. [1]
- **baseline-extractive** · `rs-10` (grenzfall) – Mein Laptop wurde um 2 Uhr nachts aus dem versperrten Kofferraum meines Mietautos gestohlen. Zahlt die Reisegepäckversicherung?
  > Aus einem unbeaufsichtigt abgestellten Fahrzeug ist Reisegepäck nur versichert, wenn es sich im versperrten Kofferraum befand und der Diebstahl zwischen 6:00 und 22:00 Uhr erfolgte. [1] Für Wertgegenstände wie Schmuck, Uhren, Kameras, Laptops und Mobiltelefone wird insgesamt höchstens 50 % der Versi
- **baseline-extractive** · `rs-11` (grenzfall) – Bin ich beim Fallschirmspringen im Urlaub versichert?
  > durch Kriegsereignisse, innere Unruhen und Terrorakte, sofern sich die versicherte Person aktiv daran beteiligt hat; [1] die die versicherte Person in einem durch Alkohol mit mehr als 0,8 Promille, Suchtgift oder Medikamentenmissbrauch beeinträchtigten Zustand verursacht hat; [1]
- **baseline-extractive** · `rs-12` (fakt) – Bis wann muss ich einen Schaden aus der Reiseversicherung melden?
  > Schäden an aufgegebenem Reisegepäck sofort beim Beförderungsunternehmen zu melden und sich eine Bestätigung (bei Flugreisen einen „Property Irregularity Report“, PIR) ausstellen zu lassen; [1] den Schaden unverzüglich, spätestens jedoch innerhalb von 3 Tagen nach Kenntnis, der Beispiel Versicherung 
- **baseline-extractive** · `kf-02` (fakt) – Ist ein Marderschaden in der Teilkasko versichert und bis zu welchem Betrag?
  > In der Teilkaskoversicherung sind Beschädigung, Zerstörung und Verlust des Fahrzeugs und seiner mitversicherten Teile versichert durch: [1] in der Teilkasko EUR 150. [4]
- **baseline-extractive** · `xd-01` (dokumentübergreifend) – Welcher Selbstbehalt gilt in der Reisegepäckversicherung und welcher in der Teilkasko?
  > Der Selbstbehalt in der Reisegepäckversicherung beträgt EUR 50 je Schadenfall. [2] In der Reisekrankenversicherung gilt kein Selbstbehalt. [4]
