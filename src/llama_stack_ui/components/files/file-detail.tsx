"use client";

import React, { useState, useEffect } from "react";
import { useParams, useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Download, Trash2, ArrowLeft, FileText } from "lucide-react";
import { useAuthClient } from "@/hooks/use-auth-client";
import { FileResource } from "@/lib/types";
import {
  formatFileSize,
  getFileTypeIcon,
  formatTimestamp,
  formatPurpose,
  isTextFile,
} from "@/lib/file-utils";
import {
  DetailLoadingView,
  DetailErrorView,
  DetailNotFoundView,
  DetailLayout,
  PropertiesCard,
  PropertyItem,
} from "@/components/layout/detail-layout";
import { CopyButton } from "@/components/ui/copy-button";

export function FileDetail() {
  const params = useParams();
  const router = useRouter();
  const client = useAuthClient();
  const fileId = params.id as string;

  const [file, setFile] = useState<FileResource | null>(null);
  const [fileContent, setFileContent] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [contentLoading, setContentLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  useEffect(() => {
    if (!fileId) return;

    const fetchFile = async () => {
      try {
        setLoading(true);
        setError(null);

        const response = await client.files.retrieve(fileId);
        setFile(response as FileResource);
      } catch (err) {
        console.error("Failed to fetch file:", err);
        setError(
          err instanceof Error ? err : new Error("Failed to fetch file")
        );
      } finally {
        setLoading(false);
      }
    };

    fetchFile();
  }, [fileId, client]);

  const handleLoadContent = async () => {
    if (!file || !isTextFile(file.filename.split(".").pop() || "")) return;

    try {
      setContentLoading(true);
      const response = await client.files.content(fileId);

      if (typeof response === "string") {
        setFileContent(response);
      } else {
        // Handle other response types
        setFileContent(JSON.stringify(response, null, 2));
      }
    } catch (err) {
      console.error("Failed to load file content:", err);
      alert("Failed to load file content");
    } finally {
      setContentLoading(false);
    }
  };

  const handleDownload = async () => {
    if (!file) return;

    try {
      const response = await client.files.content(fileId);

      let downloadUrl: string;
      let mimeType = "application/octet-stream";

      // Determine MIME type from file extension
      const extension = file.filename.split(".").pop()?.toLowerCase();
      switch (extension) {
        case "pdf":
          mimeType = "application/pdf";
          break;
        case "txt":
          mimeType = "text/plain";
          break;
        case "md":
          mimeType = "text/markdown";
          break;
        case "html":
          mimeType = "text/html";
          break;
        case "csv":
          mimeType = "text/csv";
          break;
        case "json":
          mimeType = "application/json";
          break;
        case "docx":
          mimeType =
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document";
          break;
        case "doc":
          mimeType = "application/msword";
          break;
      }

      if (typeof response === "string") {
        const blob = new Blob([response], { type: mimeType });
        downloadUrl = URL.createObjectURL(blob);
      } else if (response instanceof Blob) {
        downloadUrl = URL.createObjectURL(response);
      } else {
        const blob = new Blob([JSON.stringify(response, null, 2)], {
          type: "application/json",
        });
        downloadUrl = URL.createObjectURL(blob);
      }

      // Trigger download
      const link = document.createElement("a");
      link.href = downloadUrl;
      link.download = file.filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(downloadUrl);
    } catch (err) {
      console.error("Failed to download file:", err);
      alert("Failed to download file");
    }
  };

  const handleDelete = async () => {
    if (!file) return;

    if (
      !confirm(
        `Are you sure you want to delete "${file.filename}"? This action cannot be undone.`
      )
    ) {
      return;
    }

    try {
      setIsDeleting(true);
      await client.files.delete(fileId);
      router.push("/logs/files");
    } catch (err) {
      console.error("Failed to delete file:", err);
      alert("Failed to delete file");
    } finally {
      setIsDeleting(false);
    }
  };

  if (loading) {
    return <DetailLoadingView />;
  }

  if (error) {
    return <DetailErrorView title="File Details" id={fileId} error={error} />;
  }

  if (!file) {
    return <DetailNotFoundView title="File Details" id={fileId} />;
  }

  const isExpired = file.expires_at && file.expires_at * 1000 < Date.now();
  const fileIcon = getFileTypeIcon(file.filename.split(".").pop());
  const canPreview = isTextFile(file.filename.split(".").pop() || "");

  const mainContent = (
    <div className="space-y-6">
      {/* File Header */}
      <Card>
        <CardHeader>
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-3">
              <div className="text-2xl">{fileIcon}</div>
              <div>
                <CardTitle className="text-xl">{file.filename}</CardTitle>
                <div className="flex items-center gap-4 mt-2 text-sm text-muted-foreground">
                  <span>{formatFileSize(file.bytes)}</span>
                  <span>•</span>
                  <span>
                    {file.filename.split(".").pop()?.toUpperCase() || "Unknown"}
                  </span>
                  <span>•</span>
                  <span>{formatPurpose(file.purpose)}</span>
                </div>
              </div>
            </div>
            <div className="flex gap-2">
              <Button variant="outline" onClick={handleDownload}>
                <Download className="h-4 w-4 mr-2" />
                Download
              </Button>
              <Button
                variant="destructive"
                onClick={handleDelete}
                disabled={isDeleting}
              >
                <Trash2 className="h-4 w-4 mr-2" />
                {isDeleting ? "Deleting..." : "Delete"}
              </Button>
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* File Content Preview */}
      {canPreview && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-lg flex items-center gap-2">
                <FileText className="h-5 w-5" />
                Content Preview
              </CardTitle>
              {!fileContent && (
                <Button
                  variant="outline"
                  onClick={handleLoadContent}
                  disabled={contentLoading}
                >
                  {contentLoading ? "Loading..." : "Load Content"}
                </Button>
              )}
            </div>
          </CardHeader>
          {fileContent && (
            <CardContent>
              <div className="relative">
                <div className="absolute top-2 right-2 z-10">
                  <CopyButton
                    content={fileContent}
                    copyMessage="Copied file content to clipboard!"
                  />
                </div>
                <pre className="bg-muted p-4 rounded-lg text-sm overflow-auto max-h-96 whitespace-pre-wrap">
                  {fileContent}
                </pre>
              </div>
            </CardContent>
          )}
        </Card>
      )}

      {/* Additional Information */}
      <Card>
        <CardHeader>
          <CardTitle>File Information</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <div>
            <span className="font-medium">File ID:</span>
            <div className="flex items-center gap-2 mt-1">
              <code className="bg-muted px-2 py-1 rounded text-sm font-mono">
                {file.id}
              </code>
              <CopyButton
                content={file.id}
                copyMessage="Copied file ID to clipboard!"
              />
            </div>
          </div>

          {file.expires_at && (
            <div>
              <span className="font-medium">Status:</span>
              <div className="mt-1">
                <span
                  className={`inline-block px-2 py-1 rounded text-sm ${
                    isExpired
                      ? "bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-300"
                      : "bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-300"
                  }`}
                >
                  {isExpired ? "Expired" : "Active"}
                </span>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );

  const sidebar = (
    <div className="space-y-4">
      {/* Navigation */}
      <Card>
        <CardContent className="p-4">
          <Button
            variant="ghost"
            onClick={() => router.push("/logs/files")}
            className="w-full justify-start p-0"
          >
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Files
          </Button>
        </CardContent>
      </Card>

      {/* Properties */}
      <PropertiesCard>
        <PropertyItem label="ID" value={file.id} />
        <PropertyItem label="Filename" value={file.filename} />
        <PropertyItem label="Size" value={formatFileSize(file.bytes)} />
        <PropertyItem label="Purpose" value={formatPurpose(file.purpose)} />
        <PropertyItem
          label="Created"
          value={formatTimestamp(file.created_at)}
        />
        {file.expires_at && (
          <PropertyItem
            label="Expires"
            value={
              <span className={isExpired ? "text-destructive" : ""}>
                {formatTimestamp(file.expires_at)}
              </span>
            }
          />
        )}
      </PropertiesCard>
    </div>
  );

  return (
    <DetailLayout
      title="File Details"
      mainContent={mainContent}
      sidebar={sidebar}
    />
  );
}
