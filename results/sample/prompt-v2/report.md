# Evaluationsbericht: Modellvergleich

## Konfiguration
- Fragen: `41`
- Retrieval: `hybrid_rerank`
- Top-k: `4`
- Embedding: `jinaai/jina-embeddings-v2-base-de`
- Reranker: `jinaai/jina-reranker-v2-base-multilingual`
- Chunking: `section`
- Judge: `–`
- Prompt: `v2`

## Ergebnisse
| Modell | Genauigkeit | Fakten-Recall | Zitat korrekt | Korrekt abgelehnt | Fälschlich abgelehnt | Faithfulness (1–5) | Latenz p50 / p95 | Kosten / 1000 Fragen |
|---|---|---|---|---|---|---|---|---|
| llama3.2-3b | **85 %** | 85 % | 75 % | 100 % | 0 % | – | 27.0 s / 37.1 s | $0.00 |
| qwen2.5-3b | **68 %** | 74 % | 92 % | 60 % | 19 % | – | 25.4 s / 36.2 s | $0.00 |

**Genauigkeit**: beantwortbare Fragen enthalten alle erwarteten Fakten, unbeantwortbare Fragen werden abgelehnt.
**Faithfulness**: LLM-as-a-Judge, nur wenn mit `--judge` aktiviert.

## Genauigkeit nach Fragekategorie
| Modell | dokumentübergreifend | fakt | grenzfall | unbeantwortbar |
|---|---|---|---|---|
| qwen2.5-3b | 50 % | 95 % | 31 % | 60 % |
| llama3.2-3b | 100 % | 95 % | 62 % | 100 % |

## Fehlerbeispiele
- **qwen2.5-3b** · `hh-03` (grenzfall) – Mein Fahrrad wurde um 23 Uhr vor dem Haus gestohlen, obwohl es an einem Laternenmast angeschlossen war. Ist das versichert?
  > Dazu enthalten die vorliegenden Bedingungen keine Information. [1]
- **qwen2.5-3b** · `hh-05` (grenzfall) – Sind Schäden durch Hochwasser in der Haushaltsversicherung gedeckt?
  > Dazu enthalten die vorliegenden Bedingungen keine Information. [1]
- **qwen2.5-3b** · `hh-08` (grenzfall) – Mein Hund hat das Fahrrad meines Nachbarn beschädigt. Zahlt meine Haushaltsversicherung?
  > Nein, die Haushaltsversicherung versichert nicht Schäden an beweglichen Sachen, einschließlich Haustieren, ausgenommen von bestimmten Ausnahmen [1]. Der Hund ist nicht mitversichert, daher würde der Schaden nicht von der Versicherung abgedeckt.
- **qwen2.5-3b** · `hh-11` (grenzfall) – Ich bin vier Monate im Ausland und dort wird mein Laptop aus der Unterkunft gestohlen. Greift die Außenversicherung der Haushaltsversicherung?
  > Dazu enthalten die vorliegenden Bedingungen keine Information. [1]
- **qwen2.5-3b** · `hh-13` (grenzfall) – Bekomme ich für sehr alte Möbel den Neuwert ersetzt?
  > Dazu enthalten die vorliegenden Bedingungen keine Information. [1]
- **qwen2.5-3b** · `rs-04` (grenzfall) – Mein Koffer wurde am Flughafen 8 Stunden zu spät ausgeliefert. Was wird ersetzt?
  > Der Koffer wird bis EUR 300 je Person ersetzt, wenn der Reisegepäck zu mehr als 6 Stunden verspätet ausgeliefert wird [1].
- **qwen2.5-3b** · `rs-10` (grenzfall) – Mein Laptop wurde um 2 Uhr nachts aus dem versperrten Kofferraum meines Mietautos gestohlen. Zahlt die Reisegepäckversicherung?
  > Dazu enthalten die vorliegenden Bedingungen keine Information. [1]
- **qwen2.5-3b** · `kf-04` (grenzfall) – Jemand hat den Lack meines Autos zerkratzt. Zahlt die Teilkasko?
  > Dazu enthalten die vorliegenden Bedingungen keine Information.
- **qwen2.5-3b** · `kf-05` (fakt) – Welche Bestätigung brauche ich nach einem Zusammenstoß mit einem Reh?
  > Nach einem Zusammenstoß mit einem Reh müssen Sie eine Bestätigung der Polizei einholen [1].
- **qwen2.5-3b** · `kf-06` (grenzfall) – Wirkt sich ein Schaden in der Teilkasko auf meine Schadenfreiheitsstufe aus?
  > Dazu enthalten die vorliegenden Bedingungen keine Information. [1]
- **qwen2.5-3b** · `xd-01` (dokumentübergreifend) – Welcher Selbstbehalt gilt in der Reisegepäckversicherung und welcher in der Teilkasko?
  > Der Selbstbehalt in der Reisegepäckversicherung beträgt EUR 50 je Schadenfall [1]. In der Teilkaskoversicherung gilt kein Selbstbehalt [3].
- **qwen2.5-3b** · `un-03` (unbeantwortbar) – Wie hoch ist der Selbstbehalt in der Rechtsschutzversicherung?
  > Der Selbstbehalt in der Rechtsschutzversicherung beträgt EUR 50 je Schadenfall [1].
