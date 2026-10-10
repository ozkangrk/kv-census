# Paper writing rules (binding for all section writers)

1. NEVER type a result number by hand. Every quantitative statement uses a \kvc... macro from
   paper/generated/macros.tex (\input by main.tex already) or a number visible inside a generated
   table you \input. Counts of models etc. also come from macros. If a needed number is missing,
   tell the orchestrator; do not compute or invent it.
2. Cite ONLY keys from paper/references.bib (34 keys; the annotated list is in
   notes/related_work.md). At the end, ALL 34 keys must be cited somewhere in the paper
   (the completion gate allows <=10% uncited; aim for 0). Never add a new bib entry yourself.
3. English, academic register, concise. No marketing words. Use \emph sparingly.
4. Do NOT add \begin{document}, preamble, or \bibliography. Write only your assigned section
   files under paper/sections/. Use \label{} with your own prefix. Reference figures as
   fig:kv-trend, fig:kv-params, fig:mechanism, fig:capacity, fig:parity, fig:validation and tables
   tab:overview, tab:mechanism, tab:capacity, tab:validation (defined in generated tables).
5. Escaping: repo ids contain underscores - use \texttt{Qwen/Qwen2.5-0.5B} etc.
6. File ownership: you own ONLY the section files listed in your task prompt. Nothing else.
7. Compile-safe LaTeX: no exotic packages beyond article defaults + amsmath, booktabs, graphicx,
   hyperref, xcolor, microtype.
