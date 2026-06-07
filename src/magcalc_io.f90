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
   public :: read_disp_input, write_disp_output, read_sqw_input, write_sqw_output

contains

   !> S(Q,w) input layout (stream, native endianness):
   !>   int32 n, int32 nq, float64 S
   !>   complex128 Ud(3n,3n)
   !>   complex128 H_plus(2n,2n,nq), complex128 H_minus(2n,2n,nq)
   !>   float64 qgrid(3,nq), float64 ff(n,nq)
   subroutine read_sqw_input(path, n, nq, s, ud, hplus, hminus, qgrid, ff)
      character(len=*),              intent(in)  :: path
      integer,                       intent(out) :: n, nq
      real(real64),                  intent(out) :: s
      complex(kind=cp), allocatable, intent(out) :: ud(:, :)
      complex(kind=cp), allocatable, intent(out) :: hplus(:, :, :), hminus(:, :, :)
      real(kind=dp),    allocatable, intent(out) :: qgrid(:, :), ff(:, :)
      integer(int32) :: n32, nq32
      integer        :: u

      open(newunit=u, file=path, form='unformatted', access='stream', &
           status='old', action='read')
      read(u) n32; read(u) nq32; read(u) s
      n = int(n32); nq = int(nq32)
      allocate(ud(3*n, 3*n), hplus(2*n, 2*n, nq), hminus(2*n, 2*n, nq), &
               qgrid(3, nq), ff(n, nq))
      read(u) ud
      read(u) hplus
      read(u) hminus
      read(u) qgrid
      read(u) ff
      close(u)
   end subroutine read_sqw_input

   !> S(Q,w) output layout:
   !>   int32 nq, int32 n
   !>   float64 energies(n,nq), float64 intensities(n,nq), int32 info(nq)
   !>   complex128 K(3n,2n,nq), complex128 Kd(3n,2n,nq), complex128 evals(2n,nq)
   subroutine write_sqw_output(path, n, nq, energies, intensities, info, Kout, Kdout, evout)
      character(len=*), intent(in) :: path
      integer,          intent(in) :: n, nq
      real(kind=dp),    intent(in) :: energies(n, nq), intensities(n, nq)
      integer,          intent(in) :: info(nq)
      complex(kind=cp), intent(in) :: Kout(3*n, 2*n, nq), Kdout(3*n, 2*n, nq)
      complex(kind=cp), intent(in) :: evout(2*n, nq)
      integer(int32) :: n32, nq32, info32(nq)
      integer        :: u

      n32 = int(n, int32); nq32 = int(nq, int32); info32 = int(info, int32)
      open(newunit=u, file=path, form='unformatted', access='stream', &
           status='replace', action='write')
      write(u) nq32; write(u) n32
      write(u) energies
      write(u) intensities
      write(u) info32
      write(u) Kout
      write(u) Kdout
      write(u) evout
      close(u)
   end subroutine write_sqw_output

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
