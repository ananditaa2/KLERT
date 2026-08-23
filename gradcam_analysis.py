# ============================================================
# 24. QUANTITATIVE GRAD-CAM ANALYSIS
# ============================================================

print("\n======================================")
print("      GRAD-CAM REGION ANALYSIS")
print("======================================")


# ------------------------------------------------------------
# 24.1 Calculate activation area
# ------------------------------------------------------------

activation_pixels = np.count_nonzero(
    clean_mask
)

total_pixels = (
    clean_mask.shape[0] *
    clean_mask.shape[1]
)

activation_percentage = (
    activation_pixels /
    total_pixels
) * 100


# ------------------------------------------------------------
# 24.2 Bounding Box
# ------------------------------------------------------------

x, y, w, h = cv2.boundingRect(
    clean_mask
)


# ------------------------------------------------------------
# 24.3 Centroid
# ------------------------------------------------------------

moments = cv2.moments(
    clean_mask
)

if moments["m00"] != 0:

    centroid_x = (
        moments["m10"] /
        moments["m00"]
    )

    centroid_y = (
        moments["m01"] /
        moments["m00"]
    )

else:

    centroid_x = 0
    centroid_y = 0


# ------------------------------------------------------------
# 24.4 Mean activation
# ------------------------------------------------------------

activation_region = (
    refined_cam[clean_mask > 0]
)

if len(activation_region) > 0:

    mean_activation = (
        np.mean(activation_region)
    )

    max_activation = (
        np.max(activation_region)
    )

else:

    mean_activation = 0
    max_activation = 0


# ------------------------------------------------------------
# 24.5 Print analysis
# ------------------------------------------------------------

print(
    f"\nPrediction              : "
    f"{predicted_class}"
)

print(
    f"Confidence              : "
    f"{confidence_value:.2f}%"
)

print(
    f"Activation Area         : "
    f"{activation_percentage:.2f}%"
)

print(
    f"Activation Centroid     : "
    f"({centroid_x:.1f}, {centroid_y:.1f})"
)

print(
    f"Activation Bounding Box : "
    f"x={x}, y={y}, w={w}, h={h}"
)

print(
    f"Mean Activation         : "
    f"{mean_activation:.3f}"
)

print(
    f"Maximum Activation      : "
    f"{max_activation:.3f}"
)


# ============================================================
# 25. CREATE CONTOUR OVERLAY
# ============================================================

contour_image = (
    np.array(
        original_image.resize((224, 224))
    ).copy()
)


# Find contours
contours, _ = cv2.findContours(
    clean_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)


# Draw contours
cv2.drawContours(
    contour_image,
    contours,
    -1,
    (0, 255, 0),
    2
)


# Draw bounding box
if activation_pixels > 0:

    cv2.rectangle(
        contour_image,
        (x, y),
        (x + w, y + h),
        (255, 255, 0),
        2
    )


# Draw centroid
if activation_pixels > 0:

    cv2.circle(
        contour_image,
        (
            int(centroid_x),
            int(centroid_y)
        ),
        5,
        (255, 0, 0),
        -1
    )


# ============================================================
# 26. SAVE CONTOUR ANALYSIS
# ============================================================

contour_path = (
    "results/gradcam_region_analysis.jpg"
)

Image.fromarray(
    contour_image
).save(
    contour_path
)


# ============================================================
# 27. CREATE FINAL ANALYSIS FIGURE
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(15, 9)
)


# ------------------------------------------------------------
# ORIGINAL MRI
# ------------------------------------------------------------

axes[0, 0].imshow(
    display_image
)

axes[0, 0].set_title(
    "1. Original MRI",
    fontsize=13,
    fontweight="bold"
)

axes[0, 0].axis("off")


# ------------------------------------------------------------
# RAW GRAD-CAM
# ------------------------------------------------------------

axes[0, 1].imshow(
    raw_heatmap
)

axes[0, 1].set_title(
    "2. Raw Grad-CAM",
    fontsize=13,
    fontweight="bold"
)

axes[0, 1].axis("off")


# ------------------------------------------------------------
# REFINED HEATMAP
# ------------------------------------------------------------

axes[0, 2].imshow(
    refined_heatmap
)

axes[0, 2].set_title(
    "3. Refined Activation Map",
    fontsize=13,
    fontweight="bold"
)

axes[0, 2].axis("off")


# ------------------------------------------------------------
# FINAL OVERLAY
# ------------------------------------------------------------

axes[1, 0].imshow(
    refined_overlay
)

axes[1, 0].set_title(
    f"4. Grad-CAM Overlay\n"
    f"{predicted_class} ({confidence_value:.2f}%)",
    fontsize=13,
    fontweight="bold"
)

axes[1, 0].axis("off")


# ------------------------------------------------------------
# REGION ANALYSIS
# ------------------------------------------------------------

axes[1, 1].imshow(
    contour_image
)

axes[1, 1].set_title(
    "5. Activation Region",
    fontsize=13,
    fontweight="bold"
)

axes[1, 1].axis("off")


# ------------------------------------------------------------
# QUANTITATIVE INFORMATION
# ------------------------------------------------------------

axes[1, 2].axis("off")


analysis_text = (
    "GRAD-CAM ANALYSIS\n\n"

    f"Prediction:\n"
    f"{predicted_class.upper()}\n\n"

    f"Confidence:\n"
    f"{confidence_value:.2f}%\n\n"

    f"Activation Area:\n"
    f"{activation_percentage:.2f}%\n\n"

    f"Mean Activation:\n"
    f"{mean_activation:.3f}\n\n"

    f"Centroid:\n"
    f"({centroid_x:.1f}, "
    f"{centroid_y:.1f})\n\n"

    f"Bounding Box:\n"
    f"{w} × {h} pixels"
)


axes[1, 2].text(
    0.05,
    0.95,
    analysis_text,
    transform=axes[1, 2].transAxes,
    verticalalignment="top",
    fontsize=12
)


# ============================================================
# 28. FINAL FIGURE
# ============================================================

plt.suptitle(
    "Explainable AI Analysis using Grad-CAM",
    fontsize=18,
    fontweight="bold"
)

plt.tight_layout(
    rect=[0, 0, 1, 0.95]
)


# ============================================================
# 29. SAVE FINAL ANALYSIS
# ============================================================

analysis_path = (
    "results/gradcam_complete_analysis.png"
)

plt.savefig(
    analysis_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 30. SAVE NUMERICAL REPORT
# ============================================================

report_path = (
    "results/gradcam_analysis.txt"
)

with open(
    report_path,
    "w"
) as f:

    f.write(
        "GRAD-CAM ANALYSIS REPORT\n"
    )

    f.write(
        "========================\n\n"
    )

    f.write(
        f"Prediction: "
        f"{predicted_class}\n"
    )

    f.write(
        f"Confidence: "
        f"{confidence_value:.2f}%\n"
    )

    f.write(
        f"Activation Area: "
        f"{activation_percentage:.2f}%\n"
    )

    f.write(
        f"Mean Activation: "
        f"{mean_activation:.4f}\n"
    )

    f.write(
        f"Maximum Activation: "
        f"{max_activation:.4f}\n"
    )

    f.write(
        f"Centroid: "
        f"({centroid_x:.2f}, "
        f"{centroid_y:.2f})\n"
    )

    f.write(
        f"Bounding Box: "
        f"x={x}, y={y}, "
        f"width={w}, height={h}\n"
    )


# ============================================================
# 31. FINAL OUTPUT
# ============================================================

print("\n======================================")
print("   GRAD-CAM ANALYSIS COMPLETED")
print("======================================")

print(
    "\nPrediction:",
    predicted_class
)

print(
    f"Confidence: "
    f"{confidence_value:.2f}%"
)

print(
    f"Activation Area: "
    f"{activation_percentage:.2f}%"
)

print(
    f"Centroid: "
    f"({centroid_x:.1f}, "
    f"{centroid_y:.1f})"
)

print(
    "\nFinal Analysis saved to:"
)

print(
    os.path.abspath(
        analysis_path
    )
)

print(
    "\nRegion Analysis saved to:"
)

print(
    os.path.abspath(
        contour_path
    )
)

print(
    "\nNumerical Report saved to:"
)

print(
    os.path.abspath(
        report_path
    )
)

print(
    "\nProcess finished successfully!"
)