!> fmagcalc_sqw: standalone S(Q,w) driver.
!>   usage: fmagcalc_sqw <input.bin> <output.bin>
!> Reads the Phase-1 sqw inputs, runs the OpenMP KKd + intensity loop, and
!> writes energies, intensities, per-q info, and the K/Kd/evals intermediates
!> (the latter used by the parity test). See docs/INTERFACE.md.
program main_sqw
   use, intrinsic :: iso_fortran_env, only: real64
   use magcalc_kinds, only: dp, cp
   use magcalc_io,    only: read_sqw_input, write_sqw_output
   use magcalc_sqw,   only: run_sqw
   implicit none

   character(len=4096) :: in_path, out_path
   integer                       :: n, nq
   real(real64)                  :: s
   complex(kind=cp), allocatable :: ud(:, :), hplus(:, :, :), hminus(:, :, :)
   real(kind=dp),    allocatable :: qgrid(:, :), ff(:, :)
   real(kind=dp),    allocatable :: energies(:, :), intensities(:, :)
   integer,          allocatable :: info(:)
   complex(kind=cp), allocatable :: Kout(:, :, :), Kdout(:, :, :), evout(:, :)

   if (command_argument_count() /= 2) then
      write(*, '(a)') 'usage: fmagcalc_sqw <input.bin> <output.bin>'
      stop 2
   end if
   call get_command_argument(1, in_path)
   call get_command_argument(2, out_path)

   call read_sqw_input(trim(in_path), n, nq, s, ud, hplus, hminus, qgrid, ff)
   allocate(energies(n, nq), intensities(n, nq), info(nq))
   allocate(Kout(3*n, 2*n, nq), Kdout(3*n, 2*n, nq), evout(2*n, nq))

   call run_sqw(n, nq, s, qgrid, hplus, hminus, ud, ff, &
                energies, intensities, info, Kout, Kdout, evout)

   call write_sqw_output(trim(out_path), n, nq, energies, intensities, info, &
                         Kout, Kdout, evout)
   write(*, '(a,i0,a,i0,a)') 'fmagcalc_sqw: n=', n, ' nq=', nq, ' done'
end program main_sqw
