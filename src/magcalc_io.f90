!> Raw little-endian stream binary I/O for the Phase-1 matrix-stack handoff.
!> The Python exporter (python/export_model.py) writes the input file; this
!> reads it, and writes results back in the same byte layout. See docs/INTERFACE.md.
!>
!> Dispersion input file layout (stream, native endianness):
!>   int32         n            ! nspins
!>   int32         nq           ! number of q-points
!>   complex128    H(2n,2n,nq)  ! column-major (Fortran natural order)
!> Dispersion output file layout:
!>   int32         nq
!>   int32         n
!>   float64       energies(n,nq)
!>   int32         info(nq)
module magcalc_io
   use, intrinsic :: iso_fortran_env, only: int32, real64
   use magcalc_kinds, only: dp, cp
   implicit none
   private
   public :: read_disp_input, write_disp_output

contains

   subroutine read_disp_input(path, n, nq, h_plus)
      character(len=*),              intent(in)  :: path
      integer,                       intent(out) :: n, nq
      complex(kind=cp), allocatable, intent(out) :: h_plus(:, :, :)
      integer(int32) :: n32, nq32
      integer        :: u

      open(newunit=u, file=path, form='unformatted', access='stream', &
           status='old', action='read')
      read(u) n32
      read(u) nq32
      n = int(n32); nq = int(nq32)
      allocate(h_plus(2*n, 2*n, nq))
      read(u) h_plus
      close(u)
   end subroutine read_disp_input

   subroutine write_disp_output(path, n, nq, energies, info)
      character(len=*), intent(in) :: path
      integer,          intent(in) :: n, nq
      real(kind=dp),    intent(in) :: energies(n, nq)
      integer,          intent(in) :: info(nq)
      integer(int32) :: n32, nq32
      integer(int32) :: info32(nq)
      integer        :: u

      n32 = int(n, int32); nq32 = int(nq, int32)
      info32 = int(info, int32)
      open(newunit=u, file=path, form='unformatted', access='stream', &
           status='replace', action='write')
      write(u) nq32
      write(u) n32
      write(u) energies
      write(u) info32
      close(u)
   end subroutine write_disp_output

end module magcalc_io
