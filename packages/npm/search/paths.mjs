// Default locations of the files the search client loads at run time, for the
// npm package (the repository's web/lib/paths.mjs points at the repository
// layout instead). The SQLite build is part of this package. ONNX Runtime and
// the tokenizers are optional peer dependencies; when this package is served
// from a node_modules directory, they are found next to it. Models are not
// part of any package: by default they are fetched relative to the page
// (models/minilm-l6-v2/…, models/lateon-code-edge/…); pass modelBase or
// models to openIndex() to fetch them from elsewhere.

const here = import.meta.url;
const at = here.lastIndexOf('/node_modules/');
const nodeModules = at >= 0 ? here.slice(0, at + '/node_modules/'.length) : null;

export const sqliteModule = new URL('../index.mjs', here).href;
export const ortBase = nodeModules && `${nodeModules}onnxruntime-web/dist/`;
export const tokenizersModule = nodeModules && `${nodeModules}@huggingface/tokenizers/dist/tokenizers.min.mjs`;
export const modelBase = null;
