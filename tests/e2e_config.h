/* e2e_config.h -- the server command line an end-to-end test starts, as the YAML file the server
 * reads (core/fleet/config.h).
 *
 * The tests state their settings as flags, the spelling the option table and its messages use. The
 * server takes only `--config FILE`, so each start writes the flags as a file with the converter
 * the server's own refusal message prints, and runs the server on it.
 */
#pragma once
#include "rad_test.h"
#include "fleet/config.h"

#include <atomic>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace radtest {

/* {server, flags...} becomes {server, "--config", FILE}; FILE is a fresh name in the test's
 * temporary directory, left there for a failing case's record. */
inline std::vector<std::string> config_argv(const std::vector<std::string>& argv) {
    static std::atomic<int> seq{ 0 };
    std::vector<char*> raw;
    std::vector<std::string> store(argv);
    for (std::string& a : store) raw.push_back(a.data());
    const std::string path = tmp_path("server." + std::to_string(::getpid()) + "." +
                                      std::to_string(seq++) + ".yaml");
    std::ofstream(path) << rad::fleet::yaml_from_args((int)raw.size(), raw.data());
    return { argv.front(), "--config", path };
}

}  // namespace radtest
