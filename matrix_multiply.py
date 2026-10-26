from mpi4py import MPI
import numpy as np
comm=MPI.COMM_WORLD; rank=comm.rank; size=comm.size
n=8
a=np.arange(n*n,dtype=np.int64).reshape(n,n) if rank==0 else None
b=np.eye(n,dtype=np.int64)*2 if rank==0 else None
b=comm.bcast(b,root=0)
chunks=np.array_split(a,size,axis=0) if rank==0 else None
local=comm.scatter(chunks,root=0)
parts=comm.gather(local@b,root=0)
if rank==0:
    result=np.vstack(parts)
    np.testing.assert_array_equal(result,a@b)
    print('Distributed result matched serial reference',result.shape)
