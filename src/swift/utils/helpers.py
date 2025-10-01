from mpi4py import MPI  # isort:skip
import ezpz


def run_on_rank0(fn, *args, **kwargs):
    if ezpz.get_rank() == 0:
        fn(*args, **kwargs)
    MPI.COMM_WORLD.Barrier()


def get_ckpt_num(fpath):
    fpath = fpath.split(".pt")[-2]
    ckpt_num = fpath.split("-")[-1]
    return int(ckpt_num)
