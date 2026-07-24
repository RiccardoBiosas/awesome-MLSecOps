import { OGImageRoute } from "astro-og-canvas";
import { OG_PAGES } from "../../data/pageMeta";

export const { getStaticPaths, GET } = await OGImageRoute({
  pages: OG_PAGES,
  getImageOptions: (_path, page) => ({
    title: page.title,
    description: page.description,
    bgGradient: [[20, 33, 30], [8, 107, 88]],
    border: { color: [229, 187, 70], width: 18, side: "block-end" },
    padding: 72,
    font: {
      title: { color: [255, 255, 255], size: 74, weight: "Bold" },
      description: { color: [225, 235, 231], size: 34, weight: "Normal" },
    },
  }),
});