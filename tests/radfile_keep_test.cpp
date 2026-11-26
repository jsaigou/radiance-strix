/* radfile_keep_test.cpp -- what the load's page-cache drop keeps, and what it must not.
 *
 * The bug this guards: a container larger than RAM has its blob dropped from the page cache after
 * the load, so the cache is not left full of weight pages nothing reads again. The mapped tier is
 * the exception -- an n-gram table is read through its mapping for the engine's whole life -- and
 * dropping it means every gather after a restart faults the rows back from disk (measured: 1 519
 * vs 1 998 tok/s prefill on a 24k prompt, 103 GiB table). Two things have to be right for the
 * exception to work, and neither is visible from the outside:
 *
 *   1. the drop frees the blob MINUS the kept ranges, page-aligned so a page holding kept bytes is
 *      never dropped (rad_keep_complement);
 *   2. only addresses inside the container become kept ranges: a copy in RAM
 *      (--ngram-placement ram) or a table read from its own file (disk=DIR) is not the blob, and
 *      recording one would keep pages that belong to something else (rad_blob_range).
 *
 * The second is the easier one to get wrong silently: an offset computed from a pointer that is
 * not in the mapping is plausible arithmetic.
 */
#include "rad_test.h"
#include "format/radfile.h"

#include <cstdint>
#include <utility>
#include <vector>

using namespace rad;

namespace {

constexpr int64_t kPage = 4096;
constexpr int64_t kTables = 64 * 1024;          /* where the blob starts */
constexpr int64_t kBlob = 1 << 20;              /* one MiB of blob */
constexpr int64_t kSize = kTables + kBlob;      /* the container's size */

using Ranges = std::vector<std::pair<int64_t, int64_t>>;   /* (offset, length) */

Ranges complement(Ranges keep) {
    return rad_keep_complement(kTables, kSize, std::move(keep));
}

/* Total bytes a set of ranges covers, which is what "freed" or "kept" means to a page cache. */
int64_t covered(const Ranges& r) {
    int64_t n = 0;
    for (const auto& x : r) n += x.second;
    return n;
}

}  // namespace

/* ---- the drop's geometry ---------------------------------------------------------------- */

TEST(without_a_keep_list_the_whole_blob_is_dropped) {
    const Ranges freed = complement({});
    CHECK_EQ(freed.size(), size_t(1));
    CHECK_EQ(freed.at(0).first, kTables);
    CHECK_EQ(freed.at(0).second, kBlob);
}

TEST(a_kept_range_splits_the_drop_around_it) {
    const Ranges freed = complement({{kTables + 256 * 1024, 256 * 1024}});
    CHECK_EQ(freed.size(), size_t(2));
    CHECK_EQ(freed.at(0).first, kTables);
    CHECK_EQ(freed.at(0).second, 256 * 1024);
    CHECK_EQ(freed.at(1).first, kTables + 512 * 1024);
    CHECK_EQ(freed.at(1).second, kBlob - 512 * 1024);
    CHECK_EQ(covered(freed), kBlob - 256 * 1024);
}

TEST(a_kept_range_at_the_start_leaves_one_range_behind_it) {
    const Ranges freed = complement({{kTables, 128 * 1024}});
    CHECK_EQ(freed.size(), size_t(1));
    CHECK_EQ(freed.at(0).first, kTables + 128 * 1024);
    CHECK_EQ(freed.at(0).second, kBlob - 128 * 1024);
}

TEST(a_kept_range_at_the_end_leaves_one_range_before_it) {
    const Ranges freed = complement({{kSize - 128 * 1024, 128 * 1024}});
    CHECK_EQ(freed.size(), size_t(1));
    CHECK_EQ(freed.at(0).first, kTables);
    CHECK_EQ(freed.at(0).second, kBlob - 128 * 1024);
}

TEST(a_keep_list_covering_the_blob_drops_nothing) {
    CHECK(complement({{kTables, kBlob}}).empty());
    CHECK(complement({{kTables - kPage, kBlob + 2 * kPage}}).empty());
}

TEST(unsorted_and_overlapping_keep_ranges_are_merged) {
    /* 250..310 KiB and 300..400 KiB merge to one 150 KiB keep; both are page-aligned at the ends
     * after rounding, and the duplicate is the same range a second rank would send. */
    const Ranges freed = complement({
        {kTables + 300 * 1024, 100 * 1024},     /* second */
        {kTables + 250 * 1024, 60 * 1024},      /* overlaps and precedes it */
        {kTables + 250 * 1024, 60 * 1024},      /* the same range again, one per rank */
    });
    CHECK_EQ(freed.size(), size_t(2));
    CHECK_EQ(freed.at(0).first, kTables);
    /* The gap ends at the page boundary below the keep's start: 250 KiB is 62.5 pages, so 62
     * pages (253 952 bytes) are freed and the half page holding kept bytes stays. */
    CHECK_EQ(freed.at(0).second, 253952);
    CHECK_EQ(freed.at(1).first, kTables + 400 * 1024);          /* merged end: 250 + 150 KiB */
    CHECK_EQ(freed.at(1).second, kBlob - 400 * 1024 + 0);
    /* Freed plus kept is the blob, less the 2 KiB page the rounding gave back to the keep. */
    CHECK_EQ(covered(freed), kBlob - 150 * 1024 - 2048);
}

TEST(keep_ranges_outside_the_blob_are_clamped_not_trusted) {
    CHECK_EQ(covered(complement({{0, kTables}})), kBlob);        /* entirely in the tables */
    CHECK_EQ(covered(complement({{kSize, 4096}})), kBlob);       /* past the end */
    CHECK_EQ(covered(complement({{-4096, 4096}})), kBlob);       /* before the mapping */
    CHECK_EQ(covered(complement({{0, 0}})), kBlob);              /* empty */
    /* Half in, half out: only the part inside the blob is kept. */
    CHECK_EQ(covered(complement({{kSize - 8192, 16384}})), kBlob - 8192);
}

TEST(a_gap_never_takes_a_page_that_holds_kept_bytes) {
    /* The kept range starts four bytes into a page; the drop before it must stop at that page's
     * boundary, not one byte before the kept range, or the kept bytes share a dropped page. */
    const Ranges freed = complement({{kTables + 100 * 1024 + 4, 4096}});
    CHECK_EQ(freed.size(), size_t(2));
    CHECK_EQ(freed.at(0).second % kPage, 0);
    CHECK(freed.at(0).first + freed.at(0).second <= kTables + 100 * 1024 + 4);
    CHECK_EQ(freed.at(1).first % kPage, 0);
    CHECK(freed.at(1).first >= kTables + 100 * 1024 + 4 + 4096);
}

TEST(a_container_with_no_blob_has_nothing_to_drop_or_keep) {
    const auto none = rad_keep_complement(8192, 8192, {{8192, 4096}});
    CHECK(none.empty());
}

/* ---- which addresses are the blob -------------------------------------------------------- */

TEST(the_container_covers_its_own_bytes_and_nothing_else) {
    const std::vector<uint8_t> container(4 * kPage);
    const uint8_t* base = container.data();
    const int64_t size = (int64_t)container.size();
    int64_t off = -1, len = -1;

    CHECK(rad_blob_range(base, size, base, 16, &off, &len));
    CHECK_EQ(off, int64_t(0));
    CHECK_EQ(len, int64_t(16));

    CHECK(rad_blob_range(base, size, base + 2 * kPage + 7, 100, &off, &len));
    CHECK_EQ(off, 2 * kPage + 7);
    CHECK_EQ(len, int64_t(100));

    /* Up to and including the last byte. */
    CHECK(rad_blob_range(base, size, base + size - 1, 1, &off, &len));
    CHECK_EQ(off, size - 1);

    /* One byte past the end, or a range that overruns it. */
    CHECK(!rad_blob_range(base, size, base + size, 1, &off, &len));
    CHECK(!rad_blob_range(base, size, base + size - 1, 2, &off, &len));

    /* A copy in RAM (the --ngram-placement ram case) is not the blob. */
    const std::vector<uint8_t> copy(4 * kPage);
    CHECK(!rad_blob_range(base, size, copy.data(), 16, &off, &len));

    /* Degenerate input refuses rather than computing a range from nothing. */
    CHECK(!rad_blob_range(nullptr, size, base, 16, &off, &len));
    CHECK(!rad_blob_range(base, 0, base, 16, &off, &len));
    CHECK(!rad_blob_range(base, size, nullptr, 16, &off, &len));
    CHECK(!rad_blob_range(base, size, base, 0, &off, &len));
    CHECK(!rad_blob_range(base, size, base, -8, &off, &len));
}

TEST(the_recorded_range_is_what_the_drop_is_asked_to_keep) {
    /* End to end on the two helpers alone: map a "table" inside a container, keep it, and check
     * that the freed ranges are its complement. */
    const std::vector<uint8_t> container(kSize);
    const uint8_t* base = container.data();
    const void* table = base + kTables + 4096;
    const int64_t n = 64 * 1024;

    int64_t off = 0, len = 0;
    CHECK(rad_blob_range(base, kSize, table, n, &off, &len));
    const Ranges freed = rad_keep_complement(kTables, kSize, {{off, len}});
    CHECK_EQ(covered(freed), kBlob - n);
    for (const auto& r : freed) {
        CHECK(!(r.first < off + len && off < r.first + r.second));   /* no overlap with the keep */
    }
}

RAD_TEST_MAIN()
