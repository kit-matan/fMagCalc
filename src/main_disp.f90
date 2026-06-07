!> fmagcalc_disp: standalone dispersion driver.
!>   usage: fmagcalc_disp <input.bin> <output.bin>
!> Reads the Phase-1 matrix stack, runs the OpenMP zgeev dispersion loop, and
!> writes energies + per-q info. See docs/INTERFACE.md.
program main_disp
   use, intrinsic :: iso_fortran_env, only: real64, int64
   use magcalc_kinds,  only: dp, cp
   use magcalc_io,     only: read_disp_input, write_disp_output
   use magcalc_driver, only: run_dispersion
   implicit none

   character(len=4096) :: in_path, out_path
   integer                       :: n, nq, nargs
   complex(kind=cp), allocatable :: h_plus(:, :, :)
   real(kind=dp),    allocatable :: energies(:, :)
   integer,          allocatable :: info(:)
   integer(int64)                :: c0, c1, crate
   real(real64)                  :: compute_seconds

   nargs = command_argument_count()
   if (nargs /= 2) then
      write(*, '(a)') 'usage: fmagcalc_disp <input.bin> <output.bin>'
      stop 2
   end if
   call get_command_argument(1, in_path)
   call get_command_argument(2, out_path)

   call read_disp_input(trim(in_path), n, nq, h_plus)
   allocate(energies(n, nq), info(nq))
   call system_clock(c0, crate)
   call run_dispersion(n, nq, h_plus, energies, info)
   call system_clock(c1)
   compute_seconds = real(c1 - c0, real64) / real(crate, real64)
   call write_disp_output(trim(out_path), n, nq, energies, info)

   write(*, '(a,i0,a,i0,a,f0.6)') 'fmagcalc_disp: n=', n, ' nq=', nq, &
        ' compute_seconds=', compute_seconds
end program main_disp
