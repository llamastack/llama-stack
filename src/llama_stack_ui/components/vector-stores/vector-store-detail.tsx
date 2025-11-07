"use client";

import { useRouter } from "next/navigation";
import type { VectorStore } from "llama-stack-client/resources/vector-stores/vector-stores";
import type { VectorStoreFile } from "llama-stack-client/resources/vector-stores/files";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import {
  DetailLoadingView,
  DetailErrorView,
  DetailNotFoundView,
  DetailLayout,
  PropertiesCard,
  PropertyItem,
} from "@/components/layout/detail-layout";
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

interface VectorStoreDetailViewProps {
  store: VectorStore | null;
  files: VectorStoreFile[];
  isLoadingStore: boolean;
  isLoadingFiles: boolean;
  errorStore: Error | null;
  errorFiles: Error | null;
  id: string;
}

export function VectorStoreDetailView({
  store,
  files,
  isLoadingStore,
  isLoadingFiles,
  errorStore,
  errorFiles,
  id,
}: VectorStoreDetailViewProps) {
  const title = "Vector Store Details";
  const router = useRouter();

  const handleFileClick = (fileId: string) => {
    router.push(`/logs/vector-stores/${id}/files/${fileId}`);
  };

  if (errorStore) {
    return <DetailErrorView title={title} id={id} error={errorStore} />;
  }
  if (isLoadingStore) {
    return <DetailLoadingView title={title} />;
  }
  if (!store) {
    return <DetailNotFoundView title={title} id={id} />;
  }

  const mainContent = (
    <>
      <Card>
        <CardHeader>
          <CardTitle>Files</CardTitle>
        </CardHeader>
        <CardContent>
          {isLoadingFiles ? (
            <Skeleton className="h-4 w-full" />
          ) : errorFiles ? (
            <div className="text-destructive text-sm">
              Error loading files: {errorFiles.message}
            </div>
          ) : files.length > 0 ? (
            <Table>
              <TableCaption>Files in this vector store</TableCaption>
              <TableHeader>
                <TableRow>
                  <TableHead>ID</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Created</TableHead>
                  <TableHead>Usage Bytes</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {files.map(file => (
                  <TableRow key={file.id}>
                    <TableCell>
                      <Button
                        variant="link"
                        className="p-0 h-auto font-mono text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300"
                        onClick={() => handleFileClick(file.id)}
                      >
                        {file.id}
                      </Button>
                    </TableCell>
                    <TableCell>{file.status}</TableCell>
                    <TableCell>
                      {new Date(file.created_at * 1000).toLocaleString()}
                    </TableCell>
                    <TableCell>{file.usage_bytes}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <p className="text-gray-500 italic text-sm">
              No files in this vector store.
            </p>
          )}
        </CardContent>
      </Card>
    </>
  );

  const sidebar = (
    <PropertiesCard>
      <PropertyItem label="ID" value={store.id} />
      <PropertyItem label="Name" value={store.name || ""} />
      <PropertyItem
        label="Created"
        value={new Date(store.created_at * 1000).toLocaleString()}
      />
      <PropertyItem label="Status" value={store.status} />
      <PropertyItem label="Total Files" value={store.file_counts.total} />
      <PropertyItem label="Usage Bytes" value={store.usage_bytes} />
      <PropertyItem
        label="Provider ID"
        value={(store.metadata.provider_id as string) || ""}
      />
      <PropertyItem
        label="Provider DB ID"
        value={(store.metadata.provider_vector_db_id as string) || ""}
      />
    </PropertiesCard>
  );

  return (
    <DetailLayout title={title} mainContent={mainContent} sidebar={sidebar} />
  );
}
