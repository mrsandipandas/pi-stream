// Indoor Enclosure for Raspberry Pi Camera Rev 1.3
// Two pieces: housing + snap-on lid.
// Lens faces forward through collar. Board slides in from top.
// Lid snaps on and has a cable slot for the ribbon cable.
// Screw holes in base flange for mounting.

/* ── Camera board ── */
board_w      = 25;
board_h      = 24;
board_t      = 1.2;
hole_spacing = 21;
hole_dia     = 2.2;

lens_dia     = 8.5;      // lens barrel diameter
lens_h       = 5.0;      // lens height above PCB
lens_cx      = 0;        // lens X offset from board center
lens_cy      = 2.0;      // lens Y offset — ABOVE center, toward connector

/* ── Enclosure ── */
wall         = 2.0;
clearance    = 0.6;
cable_w      = 16.5;     // ribbon cable width
cable_slot_h = 3;        // cable slot height in lid
corner_r     = 2;

// Lens collar
lens_hole_dia = lens_dia + 0.6;   // 9.1mm bore
collar_od    = lens_dia + 4;      // 12.5mm outer
collar_protrude = 3;              // sticks out front

// Base flange screws
flange_screw_dia = 3.2;  // M3 clearance

// Lid snap-fit
snap_tol     = 0.3;      // tolerance for snap fit
snap_depth   = 1.5;      // how deep the lip sits
lip_h        = 2.5;      // inner lip height on lid

/* ── Derived ──
   Board upright: X=width, Z=depth(front-back), Y=height
   Front = -Z, lens points -Z
*/
inner_x = board_w + 2 * clearance;
inner_z = board_t + lens_h + 2 * clearance;
inner_y = board_h + 2 * clearance;
outer_x = inner_x + 2 * wall;
outer_z = inner_z + 2 * wall;
outer_y = inner_y + wall;   // walls on bottom + sides, lid closes top

flange_x = outer_x + 10;
flange_z = outer_z + 6;

// Lens Y position from base
lens_y = wall + clearance + board_h/2 + lens_cy;

lid_outer_y = wall;     // lid thickness

/* ══════════════════════════════
   Housing — open top, board slides in
   ══════════════════════════════ */
module camera_housing() {
    difference() {
        union() {
            // Main body — open top box
            difference() {
                rounded_box(outer_x, outer_z, outer_y, corner_r);
                // Hollow inside (open top)
                translate([0, 0, wall + inner_y/2])
                    cube([inner_x, inner_z, inner_y + wall + 1], center=true);
            }

            // Base flange
            translate([0, 0, wall/4])
                rounded_box(flange_x, flange_z, wall/2, corner_r + 2);

            // Lens collar — protrudes out front
            translate([lens_cx, -(outer_z/2), lens_y])
                lens_collar();
        }

        // Square lens bore through front wall + collar
        translate([lens_cx, -(outer_z/2 + collar_protrude + 1), lens_y])
            rotate([-90, 0, 0])
                cube([lens_hole_dia, lens_hole_dia, wall + collar_protrude + 2], center=true);

        // Flange screw holes (4 corners)
        for (x = [-(flange_x/2 - 4), (flange_x/2 - 4)])
            for (y = [-(flange_z/2 - 4), (flange_z/2 - 4)])
                translate([x, y, -1])
                    cylinder(d=flange_screw_dia, h=wall + 2, $fn=20);

        // Countersinks
        for (x = [-(flange_x/2 - 4), (flange_x/2 - 4)])
            for (y = [-(flange_z/2 - 4), (flange_z/2 - 4)])
                translate([x, y, -0.1])
                    cylinder(d1=flange_screw_dia + 3, d2=flange_screw_dia, h=1.5, $fn=20);
    }

    // PCB guide rails inside
    for (side = [-1, 1])
        translate([side * (inner_x/2 - 0.8), (clearance + board_t/2 - inner_z/2), wall + inner_y/2])
            cube([1.5, 1.2, inner_y - 4], center=true);

    // Board support shelf at bottom
    translate([0, (clearance + board_t/2 - inner_z/2), wall + 1])
        cube([inner_x - 6, 1.5, 2], center=true);
}

/* ══════════════════════════════
   Lid — snaps on top, has cable slot
   ══════════════════════════════ */
module camera_lid() {
    // Cap: Z=0 to Z=wall (flat top surface)
    // Lip: Z=-lip_h to Z=0 (hangs into housing)
    difference() {
        union() {
            // Flat cap
            rounded_box(outer_x, outer_z, wall, corner_r);

            // Lip hanging below cap
            translate([0, 0, -lip_h/2])
                cube([inner_x - 2*snap_tol, inner_z - 2*snap_tol, lip_h], center=true);
        }

        // Cable slot — through back edge of cap + lip
        translate([0, (outer_z/2 - wall/2), 0])
            cube([cable_w + 1, wall + 1, lip_h + wall + 2], center=true);
    }

    // Snap bumps on the lip
    for (yy = [-(inner_z/2 - snap_tol - 0.3), (inner_z/2 - snap_tol - 0.3)])
        translate([0, yy, -lip_h + 0.8])
            cube([inner_x/2, 0.6, 1.5], center=true);
}

/* ── Lens collar ── */
module lens_collar() {
    rotate([-90, 0, 0])
    difference() {
        cube([collar_od, collar_od, collar_protrude + wall], center=true);
        translate([0, 0, -0.1])
            cube([lens_hole_dia, lens_hole_dia, collar_protrude + wall + 0.2], center=true);
    }
}

/* ── Helper ── */
module rounded_box(w, h, d, r) {
    rr = min(r, min(w, min(h, d)) / 2 - 0.1);
    translate([0, 0, d/2])
        minkowski() {
            cube([w - 2*rr, h - 2*rr, d - 0.5], center=true);
            cylinder(r=rr, h=0.25, $fn=24);
        }
}

/* ══════════════════════════════
   RENDER
   ══════════════════════════════ */

// Preview — lid on top of housing
//color("SlateGray") camera_housing();
//color("Blue") translate([0, 0, board_w+lip_h]) camera_lid();


// For STL export, render ONE at a time:
camera_housing(); // → camera_housing.stl
//camera_lid();     // → camera_lid.stl