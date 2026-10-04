// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Eke Hive Authors
#pragma once
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <poll.h>
#include <unistd.h>
#include <cerrno>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <stdexcept>
#include <thread>
#include <vector>

#include "hive/clock.h"

namespace hive {
// Owns accepted fds until the producer calls finish AND queued bytes are drained.
// A failed peer's fd is retained until finish, preventing fd-reuse/ABA while an
// engine request still holds the integer. No inference deadline is imposed.
class SocketWriter {
  struct Peer { std::deque<std::string> queue; size_t offset=0, bytes=0; bool finished=false, failed=false;
                std::chrono::steady_clock::time_point finished_at; };
  std::mutex mu_;
  std::map<int, Peer> peers_;
  int wake_ = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  bool stop_=false;
  size_t max_bytes_;
  int drain_ms_;
  std::thread worker_;
  void wake() { uint64_t one=1; ssize_t n; do { n=::write(wake_, &one, sizeof(one)); } while(n<0 && errno==EINTR); }
  void loop() {
    for (;;) {
      std::vector<pollfd> fds{{wake_, POLLIN, 0}};
      int timeout=-1;
      {
        std::lock_guard<std::mutex> lock(mu_);
        if (stop_) return;
        for (auto it=peers_.begin(); it!=peers_.end();) {
          auto& p=it->second;
          if(p.finished && drain_ms_>0) {
            const auto elapsed=std::chrono::duration_cast<std::chrono::milliseconds>(hive::SteadyClock::now()-p.finished_at).count();
            const int left=std::max(0,drain_ms_-(int)elapsed);
            if(!left) p.failed=true;
            timeout=timeout<0?left:std::min(timeout,left);
          }
          if (p.finished && (p.failed || p.queue.empty())) { ::close(it->first); it=peers_.erase(it); continue; }
          if (!p.failed && !p.queue.empty()) fds.push_back({it->first, POLLOUT, 0});
          ++it;
        }
      }
      if (::poll(fds.data(), fds.size(), timeout)<0) continue;
      if (fds[0].revents) { uint64_t count; ssize_t n; do { n=::read(wake_, &count, sizeof(count)); } while(n<0 && errno==EINTR); }
      std::lock_guard<std::mutex> lock(mu_);
      for (size_t i=1; i<fds.size(); ++i) {
        if (!fds[i].revents) continue;
        auto it=peers_.find(fds[i].fd);
        if (it==peers_.end()) continue;
        auto& p=it->second;
        if (p.failed || p.queue.empty()) continue;
        auto& data=p.queue.front();
        const ssize_t n=::send(it->first, data.data()+p.offset, data.size()-p.offset, MSG_NOSIGNAL | MSG_DONTWAIT);
        if (n>0) { p.offset+=(size_t)n; p.bytes-=(size_t)n; if (p.offset==data.size()) { p.queue.pop_front(); p.offset=0; } }
        else if (n==0 || (errno!=EAGAIN && errno!=EWOULDBLOCK && errno!=EINTR)) { p.failed=true; p.queue.clear(); p.bytes=0; }
      }
    }
  }
 public:
  // test_host_cpu: a non-reading socket with a 4KiB send buffer would stall a
  // blocking write_all at 4MiB. Bound queued output, not generation time;
  // after terminal finish allow 30s to drain, then reclaim the abandoned fd.
  explicit SocketWriter(size_t max_bytes=8u<<20,int drain_ms=30000):max_bytes_(max_bytes),drain_ms_(drain_ms) {
    if(wake_<0) throw std::runtime_error("eventfd unavailable"); worker_=std::thread([this]{loop();});
  }
  ~SocketWriter() {
    { std::lock_guard<std::mutex> lock(mu_); stop_=true; }
    wake(); worker_.join();
    for (auto& [fd,p]:peers_) ::close(fd);
    ::close(wake_);
  }
  void add(int fd) { std::lock_guard<std::mutex> lock(mu_); peers_.emplace(fd, Peer{}); }
  bool send(int fd, std::string data) {
    { std::lock_guard<std::mutex> lock(mu_);
      auto it=peers_.find(fd);
      if (it==peers_.end() || it->second.failed || it->second.finished) return false;
      if(data.size()>max_bytes_ || it->second.bytes>max_bytes_-data.size()) {
        it->second.failed=true;it->second.queue.clear();it->second.bytes=0;wake();return false;
      }
      // If the queue is empty, try one non-blocking send on the engine thread so that a vanished peer
      //   (EPIPE/ECONNRESET) is reported by this call itself; otherwise the writer thread would only notice it later
      //   and the engine would run one more prefill chunk. An unsent remainder or EAGAIN is queued (order preserved).
      Peer& p=it->second;
      if (p.queue.empty()) {
        ssize_t n;
        do { n=::send(fd, data.data(), data.size(), MSG_NOSIGNAL | MSG_DONTWAIT); } while(n<0 && errno==EINTR);
        if (n<0 && errno!=EAGAIN && errno!=EWOULDBLOCK) { p.failed=true; p.bytes=0; wake(); return false; }  // same rule as loop()
        if (n>=0 && (size_t)n==data.size()) return true;
        if (n>0) data.erase(0,(size_t)n);
      }
      p.bytes+=data.size();
      p.queue.push_back(std::move(data));
    }
    wake(); return true;
  }
  // Graceful stop (hived HIVE_GRACEFUL_STOP_S): wait until finished connections have flushed their remaining bytes
  //   and closed (bounded by ms). Returns true when no connection is left. The destructor drops queues and closes
  //   immediately, so call this first when the final messages must be delivered.
  bool wait_idle(int ms) {
    const auto until=hive::SteadyClock::now()+std::chrono::milliseconds(ms);
    for (;;) {
      { std::lock_guard<std::mutex> lock(mu_); if (peers_.empty()) return true; }
      if (std::chrono::steady_clock::now()>=until) return false;
      wake(); std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  }
  void finish(int fd) {
    { std::lock_guard<std::mutex> lock(mu_); auto it=peers_.find(fd); if(it!=peers_.end() && !it->second.finished) {
        it->second.finished=true;it->second.finished_at=std::chrono::steady_clock::now(); } }
    wake();
  }
};
} // namespace hive
