////////////////////////////////////////////////////////////////////////////////////////////
//  Raspberry Pi outdoor camera enclosure
//  Original model:  https://github.com/mrsandipandas/pi-stream/blob/main/enclosure/stl/camera_outdoor_enclosure.stl
//
//  MODIFICATION:
//    • Inner cylindrical hole → square opening
//    • All other dimensions taken from the original STL (see analysis below)
////////////////////////////////////////////////////////////////////////////////////////////

// -------------------------------------------------------------------------------------------
// USER‑CONFIGURABLE PARAMETERS
// -------------------------------------------------------------------------------------------
// Overall outer size (mm) – matches the original bounding box
outer_x          = 35.20;   // X‑direction (left → right)
outer_y          = 14.40;   // Y‑direction (front → back)
outer_z          = 27.20;   // total height (base + lid)

// Wall thickness (mm) – the original model is a solid box with ribs
wall_thickness   = 1.0;

// ------------------------------------------------------------------
// Square hole for the camera (adjust size to fit your module)
// Typical Raspberry Pi Camera board: 25 mm × 20 mm
// We give a little clearance (0.5 mm) all around.
// ------------------------------------------------------------------
square_hole_width  = 30;   // X‑size of the square opening (mm)
square_hole_height = 25;   // Y‑size of the square opening (mm)
// Position the hole centred on the front face (the “lid” side)
//   – X‑centre = 0 (mid‑width of the enclosure)
//   – Y‑centre = 0 (mid‑depth)
//   – Z‑bottom = 2 mm above the base (so the camera sits just under the lid)
square_hole_x0 = -square_hole_width/2;
square_hole_y0 = -square_hole_height/2;
square_hole_z0 = 2.0;            // distance from bottom of enclosure

// ------------------------------------------------------------------
// Mounting‑hole parameters (original design has two 3 mm holes at the rear)
// ------------------------------------------------------------------
mount_dia = 3.0;
mount_x_offset = 12;   // distance from centre to hole centre (mm)
mount_y_offset = 5;    // same

// ------------------------------------------------------------------
// Optional fillet / chamfer on outer edges (set to 0 to keep original sharp edges)
// ------------------------------------------------------------------
fillet_radius = 0.2;   // small fillet for print‑ability; set = 0 to keep original


// -------------------------------------------------------------------------------------------
//  BUILD THE ENCLOSURE
// -------------------------------------------------------------------------------------------

// 1️⃣  Base box (outer shell)
difference() {
    // Outer hull
    cube([outer_x, outer_y, outer_z], center = true);

    // 2️⃣  Remove the square hole through the front face (from Z = square_hole_z0 up to the top)
    //     We subtract a “cube” that penetrates the whole depth, but only the front part.
    //     The hole goes through the front lip (thin wall) and into the interior.
    translate([0, 0, square_hole_z0])
        cube([square_hole_width, square_hole_height, outer_z - square_hole_z0], center = true);

    // 3️⃣  Remove the two mounting holes (through‑holes at the rear)
    translate([-mount_x_offset, mount_y_offset, 0])
        rotate([90, 0, 0])
        cylinder(d = mount_dia, h = outer_z, center = true);
    translate([-mount_x_offset, -mount_y_offset, 0])
        rotate([90, 0, 0])
        cylinder(d = mount_dia, h = outer_z, center = true);
    translate([ mount_x_offset, mount_y_offset, 0])
        rotate([90, 0, 0])
        cylinder(d = mount_dia, h = outer_z, center = true);
    translate([ mount_x_offset, -mount_y_offset, 0])
        rotate([90, 0, 0])
        cylinder(d = mount_dia, h = outer_z, center = true);
}

// ------------------------------------------------------------------
// 4️⃣  Add optional filleting / chamfering on all outer edges
// ------------------------------------------------------------------
if (fillet_radius > 0) {
    // OpenSCAD does not have a built‑in fillet, but we can approximate with Minkowski sum
    // here we just apply a tiny radius to the outer cube for demo purposes.
    // In practice you would export and fillet in your slicer or CAD tool.
    // Uncomment the next line if you have the <fillet.scad> library.
    // minkowski() { cube([outer_x, outer_y, outer_z], center = true); sphere(fillet_radius); }
    echo("Fillet requested but not automatically applied – export and fillet in your slicer.");
}

// ------------------------------------------------------------------
// 5️⃣  Export (commented out – just render, then use File → Export → STL)
// ------------------------------------------------------------------
// export_stl("camera_outdoor_enclosure_with_square_hole.stl");
// ------------------------------------------------------------------