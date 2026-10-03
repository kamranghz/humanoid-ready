"""Windows-safe ``torchrun`` entry (libuv-free TCPStore for elastic rendezvous)."""

from __future__ import annotations

import os

os.environ["USE_LIBUV"] = "0"
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")


def _patch_static_tcp_rendezvous() -> None:
    """PyTorch elastic StaticTCPRendezvous ignores USE_LIBUV; pass use_libuv=False."""
    from torch.distributed import PrefixStore, TCPStore
    from torch.distributed.elastic.rendezvous import RendezvousInfo, RendezvousStoreInfo
    import torch.distributed.elastic.rendezvous.static_tcp_rendezvous as mod

    def next_rendezvous(self) -> RendezvousInfo:
        mod.logger.info("Creating TCPStore (use_libuv=False) for Windows/gloo")
        is_master = self.rank == 0
        if not self._store:
            self._store = TCPStore(  # type: ignore[call-arg]
                self.master_addr,
                self.master_port,
                self.world_size,
                is_master,
                self.timeout,
                multi_tenant=True,
                use_libuv=False,
            )
        store = PrefixStore(self.run_id, self._store)
        bootstrap = RendezvousStoreInfo(self.master_addr, self.master_port)
        return RendezvousInfo(store, self.rank, self.world_size, bootstrap)

    mod.StaticTCPRendezvous.next_rendezvous = next_rendezvous  # type: ignore[method-assign]


if __name__ == "__main__":
    _patch_static_tcp_rendezvous()
    from torch.distributed.run import main

    main()
