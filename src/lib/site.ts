/**
 * Site-wide constants: this site's URL, its companion sites, the owner's
 * slide series it links into, and its repository.
 */

/** This site (production). */
export const SITE_URL = "https://gpu-kernels-explained.vercel.app";

/** The companion sites. */
export const DECODER_URL = "https://transformer-decoder-explained.vercel.app";
export const INFERENCE_URL = "https://llm-inference-explained.vercel.app";
export const ARCHITECTURES_URL =
  "https://llm-architectures-explained.vercel.app";
export const NUMERICS_URL = "https://numerics-explained.vercel.app";
export const SILICON_URL = "https://systolic-arrays-explained.vercel.app";
export const TRADEOFFS_URL = "https://inference-tradeoffs-explained.vercel.app";

export const GITHUB_URL =
  "https://github.com/BrendanJamesLynskey/gpu-kernels-explained";

/** The slide series each chapter links into ("go deeper"). */
export const NVIDIA_GPU_HUB =
  "https://brendanjameslynskey.github.io/LLM_Hub_NVIDIA_GPUs/";
export const CUDA_HUB = "https://brendanjameslynskey.github.io/LLM_Hub_CUDA/";

/** A slide in one of the owner's decks (anchors are #slide-NN). */
export function deck(repo: string, slide?: number): string {
  const base = `https://brendanjameslynskey.github.io/${repo}/`;
  return slide === undefined
    ? base
    : `${base}#slide-${String(slide).padStart(2, "0")}`;
}

/** A file in this site's repository on GitHub. */
export function repoFile(path: string): string {
  return `${GITHUB_URL}/blob/main/${path}`;
}
