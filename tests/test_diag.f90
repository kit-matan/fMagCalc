!> Smoke test: zgeev recovers known eigenvalues of a 2x2 complex matrix.
program test_diag
   use magcalc_kinds, only: dp, cp
   use magcalc_diag,  only: eig_zgeev
   implicit none

   complex(kind=cp) :: a(2, 2), w(2), vr(2, 2)
   integer          :: info
   real(kind=dp)    :: lo, hi
   real(kind=dp), parameter :: tol = 1.0e-10_dp

   ! Diagonal matrix diag(2, 5): eigenvalues are exactly 2 and 5.
   a = reshape([(2.0_dp, 0.0_dp), (0.0_dp, 0.0_dp), &
                (0.0_dp, 0.0_dp), (5.0_dp, 0.0_dp)], [2, 2])

   call eig_zgeev(2, a, w, vr, info)
   if (info /= 0) then
      write (*, *) 'FAIL: zgeev info =', info
      stop 1
   end if

   lo = min(real(w(1), dp), real(w(2), dp))
   hi = max(real(w(1), dp), real(w(2), dp))
   if (abs(lo - 2.0_dp) > tol .or. abs(hi - 5.0_dp) > tol) then
      write (*, *) 'FAIL: eigenvalues =', w
      stop 1
   end if

   write (*, *) 'PASS: test_diag'
end program test_diag
