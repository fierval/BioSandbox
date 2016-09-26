#load "load-project-release.fsx"
open DrawGraph

createGraph "digraph{a->b; b->c; 2->1; d->b; b->b; a->d}" "dot.exe" None

