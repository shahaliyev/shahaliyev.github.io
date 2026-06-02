# Derive URL slug from dated filenames (e.g. 2025-08-25-kalokagathia -> kalokagathia).
Jekyll::Hooks.register :documents, :post_init do |doc|
  next unless doc.relative_path.start_with?("_notes/")

  basename = File.basename(doc.relative_path, ".*")
  slug = doc.data["slug"]
  if slug.nil? || slug.to_s.strip.empty?
    slug = basename.sub(/\A\d{4}-\d{2}-\d{2}-/, "")
    doc.data["slug"] = slug
  end

  label = doc.data["label"]
  if label && !label.to_s.strip.empty?
    doc.data["title"] = label.to_s.strip
  elsif doc.data["title"].nil? || doc.data["title"].to_s.strip.empty?
    doc.data["title"] = slug.tr("-", " ").split.map(&:capitalize).join(" ")
  end
end
