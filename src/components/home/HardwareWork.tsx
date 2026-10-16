/**
 * "Built by a hardware engineer": the author's own RTL, SoC, signal-integrity
 * and FPGA work, linked from every Hardware-track landing page (brief 42).
 * The same file on Kernels, Systolic, Accelerators and Racks. The decoder's
 * figures are quoted from its README's Vivado table (they are the author's
 * synthesis runs, not this site's). Server Component, no client code.
 */
const LINK =
  "focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent";

const GH = "https://github.com/BrendanJamesLynskey";
const PAGES = "https://brendanjameslynskey.github.io";

const ITEMS: { href: string; label: string; text: string }[] = [
  {
    href: `${GH}/LLM_Transformer_Decoder_RTL`,
    label: "A transformer decoder in RTL",
    text: "SystemVerilog with 84 passing tests (a Python golden model and cocotb), synthesised in Vivado 2025.2 for an Artix-7: its 4 × 4 systolic array is 455 LUTs, 901 flip-flops and 16 DSPs at 156.8 MHz (the README's synthesis table).",
  },
  {
    href: `${PAGES}/LLM_Transformer_Decoder_RTL/`,
    label: "The decoder, module by module",
    text: "A walkthrough of every RTL module, the verification and the synthesis numbers.",
  },
  {
    href: `${GH}/Interview_RTL_LLM_Accelerators`,
    label: "RTL challenges for LLM accelerators",
    text: "Systolic array, mixed-precision MAC, softmax pipeline, KV cache and more, each simulated (two run inside Systolic Arrays Explained).",
  },
  {
    href: `${GH}/Interview_FPGA`,
    label: "FPGA design challenges",
    text: "FPGA architecture, synthesis, timing closure and the Vivado and Quartus flows, with RTL exercises.",
  },
  {
    href: `${GH}/RISCV_RV32IMC_5stage`,
    label: "A five-stage RISC-V SoC",
    text: "BRV32P: a pipelined RV32IMC core with caches and an AXI4-Lite fabric in Verilog, and a MESI cache controller (",
  },
  {
    href: `${PAGES}/SoC/`,
    label: "Modern SoC design",
    text: "Slide decks from chiplets and advanced packaging to on-chip interconnects and silicon implementation.",
  },
  {
    href: `${PAGES}/SerDes_Equalisation/`,
    label: "Equalisation in high-speed serial links",
    text: "One 28 GBd backplane channel end to end: S-parameters, CTLE, FFE and DFE, a noise budget.",
  },
  {
    href: `${PAGES}/Signal_Integrity/`,
    label: "Signal integrity",
    text: "Seventeen decks on getting a signal from one chip to another intact, every number computed by a model.",
  },
  {
    href: `${PAGES}/Hardware/`,
    label: "All the hardware decks",
    text: "RISC-V, Arm, AMBA, simulators, photonics and more.",
  },
];

export function HardwareWork(): JSX.Element {
  return (
    <section
      aria-labelledby="hw-work-heading"
      className="mt-12 rounded-lg border border-neutral-200 p-5 dark:border-neutral-800"
      data-hardware-work
    >
      <h2 id="hw-work-heading" className="text-lg font-semibold tracking-tight">
        Built by a hardware engineer
      </h2>
      <p className="mt-2 max-w-3xl text-sm text-neutral-600 dark:text-neutral-400">
        The author designs hardware: the RTL, SoC and signal-integrity work
        behind this track is public too.
      </p>
      <ul className="mt-4 grid gap-3 text-sm sm:grid-cols-2">
        {ITEMS.map((it) => (
          <li key={it.href} className="min-w-0">
            <a href={it.href} className={`${LINK} font-medium`}>
              {it.label}
            </a>
            <span className="text-neutral-600 dark:text-neutral-400">
              {": "}
              {it.text}
              {it.href.endsWith("RISCV_RV32IMC_5stage") ? (
                <>
                  <a href={`${GH}/Cache_Controller_MESI`} className={LINK}>
                    Cache_Controller_MESI
                  </a>
                  ).
                </>
              ) : null}
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}
