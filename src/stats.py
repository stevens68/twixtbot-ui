import pstats
from pstats import SortKey
p = pstats.Stats('profile_output')
# p.strip_dirs().sort_stats(-1).print_stats()

print("--------------------------  CALLS")
p.sort_stats(SortKey.CALLS).print_stats(10)

print("--------------------------  CUMULATIVE")
p.sort_stats(SortKey.CUMULATIVE).print_stats(10)

print("--------------------------  FILENAME")
p.sort_stats(SortKey.FILENAME).print_stats(10)

print("--------------------------  PCALLS")
p.sort_stats(SortKey.PCALLS).print_stats(10)

print("--------------------------  TIME")
p.sort_stats(SortKey.TIME).print_stats(10)

print("--------------------------  NFL")
p.sort_stats(SortKey.NFL).print_stats(10)
