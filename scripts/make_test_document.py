"""Generate claim_aware_rag_test_document.pdf (a SYNTHETIC test document).

The content is invented for testing retrieval, verification and abstention.
It does not describe real experiments. Requires `fpdf2` (dev dependency).

    python scripts/make_test_document.py
"""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "claim_aware_rag_test_document.pdf"

TITLE = "Field Evaluation of Low-Power Environmental Sensor Nodes"

# Each inner list is one page: (kind, text) with kind "h" (heading) or "p" (paragraph).
PAGES = [
    [
        ("p", "Synthetic test document. This report was written to test the Claim-Aware Adaptive RAG "
              "system and does not describe real experiments."),
        ("h", "1. Introduction"),
        ("p", "The study compared two deployment configurations of a battery-powered environmental sensor node. "
              "Each node measured air temperature, relative humidity and particulate matter. "
              "The goal was to understand how configuration choices affect energy consumption and data quality. "
              "Both trials used the same hardware model, the same 3000 mAh lithium battery and the same firmware version."),
        ("h", "2. Methods"),
        ("p", "Trial One sampled the sensors at a frequency of 1 Hz. "
              "Trial Two sampled the sensors at a frequency of 10 Hz. "
              "Each trial ran for 14 days at the same rooftop site. "
              "Trial One ran from March 3 to March 16, and Trial Two ran from March 17 to March 30. "
              "Two sensor nodes, Node A and Node B, were deployed in each trial. "
              "Data were transmitted over LoRaWAN every 5 minutes in both trials. "
              "Energy consumption was measured with an inline power monitor attached to each node."),
    ],
    [
        ("h", "3. Results"),
        ("p", "Trial One consumed an average of 42 milliwatt-hours per day. "
              "Trial Two consumed an average of 118 milliwatt-hours per day. "
              "Trial Two therefore consumed about 2.8 times as much energy as Trial One. "
              "The higher energy consumption in Trial Two was caused by its higher sampling frequency, "
              "which kept the microcontroller awake for longer periods. "
              "Radio transmission energy was nearly identical in both trials because the transmission interval was the same."),
        ("p", "The battery in Trial Two was projected to last 25 days, compared with 71 days in Trial One. "
              "Data quality was similar in both trials. "
              "The mean absolute temperature error was 0.3 degrees Celsius in Trial One and 0.2 degrees Celsius in Trial Two. "
              "Trial Two recorded 12 short pollution spikes lasting less than 5 seconds, whereas Trial One recorded none."),
        ("p", "Node B in Trial Two rebooted twice during a thunderstorm on March 21. "
              "The reboots were caused by a loose power connector rather than by the sampling configuration."),
    ],
    [
        ("h", "4. Discussion"),
        ("p", "Higher sampling frequency improved the detection of short pollution events. "
              "However, the improvement in temperature accuracy was small and may not justify the additional energy cost. "
              "Humidity readings during rain were excluded from the analysis because the humidity sensor saturated above 95 percent relative humidity. "
              "The rooftop site was chosen because it had an unobstructed line of sight to the gateway."),
        ("h", "5. Limitations"),
        ("p", "The study used only one site and two nodes per trial, so the results may not generalize to other environments. "
              "Ambient temperature differed between the two trial periods, which could have affected battery performance. "
              "The purchase cost of the sensor nodes was not recorded in this study."),
        ("h", "6. Conclusion"),
        ("p", "For long-term deployments where battery life is the priority, a 1 Hz sampling frequency is recommended. "
              "A 10 Hz sampling frequency is recommended only when short-lived pollution events must be captured."),
    ],
]


def build(output: Path = OUTPUT) -> Path:
    pdf = FPDF(format="A4")
    pdf.set_margins(20, 20, 20)
    for number, page in enumerate(PAGES):
        pdf.add_page()
        if number == 0:
            pdf.set_font("Helvetica", "B", 15)
            pdf.multi_cell(0, 8, TITLE, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)
        for kind, text in page:
            if kind == "h":
                pdf.ln(2)
                pdf.set_font("Helvetica", "B", 12)
                pdf.multi_cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
            else:
                pdf.set_font("Helvetica", "", 11)
                pdf.multi_cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")
                pdf.ln(2)
    pdf.output(str(output))
    return output


if __name__ == "__main__":
    print(f"Wrote {build()}")
